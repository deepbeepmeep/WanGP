# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# Drafts from repeated text and the choice of a verification window, for any model that verifies drafts, with ideas of Strata (MIT License, https://github.com/Niko1221/Strata).
"""When a text repeats (an edit of code, a quotation), what followed its earlier occurrence is a free draft: no weights, no GPU. A verification checks these drafts like those of a drafter, so
the output does not depend on them. The measured time of a verification by rows and the measured acceptance decide whether a window pays."""

MATCH_BUCKETS = (6, 12, 24)  # match lengths: the longer, the likelier its continuation
PRIOR_RATE = (0.75, 0.88, 0.93, 0.96)  # the rate of a bucket before its measures, worth PRIOR_WEIGHT of them
PRIOR_WEIGHT = 4.0
FADE = 0.97  # older windows fade from the rates
COST_WEIGHT = 0.1  # of a new measure, in the time of a window
SHAPE_PER_ROW = 0.45  # the time of a row against the first, for the sizes not measured yet
TOKENS_WEIGHT = 0.05
CONFIDENCE_FADE = 0.995  # the acceptance of a drafter's confidences fades slowly
CONFIDENCE_BUCKETS = 8
PASS_START, PASS_WEIGHT, PASS_SPIKE = 8, 0.1, 1.5  # the time of a pass: the fastest of its first measures, then a new one weighs this, counted for at most this many times the time known (other work on the GPU)


class SuffixDrafter:
    """The tokens that followed the longest earlier occurrence of a text's last tokens. An index of the trigrams of the text gives their WAYS latest places; each is extended backwards up to max_match,
    the longest wins, the latest among equals. Bounded by the context of the sequence, about 200 bytes per token."""
    WAYS = 4

    def __init__(self, min_match=3, max_match=64):
        self.min_match, self.max_match = max(3, min_match), max_match
        self.history, self.index = [], {}

    def reset(self):
        self.history, self.index = [], {}

    def _append(self, tokens):
        history, index, ways = self.history, self.index, self.WAYS
        for token in tokens:
            history.append(token)
            end = len(history) - 1
            if end >= 2:
                key = (history[end - 2] << 40) | (history[end - 1] << 20) | token
                places = index.get(key)
                if places is None:
                    index[key] = [end]
                else:
                    places.insert(0, end)
                    del places[ways:]

    def sync(self, token_ids):
        """Follows the sequence's tokens: appends the new ones, starts again after a rewind."""
        known = len(self.history)
        if known > len(token_ids) or (known and self.history[-1] != token_ids[known - 1]):
            self.reset()
            known = 0
        if known < len(token_ids):
            self._append(token_ids[known:])

    def propose(self, count):
        """Up to count tokens and the length of their match, ([], 0) without a match of min_match tokens."""
        history = self.history
        current = len(history) - 1
        if current < 3 or count <= 0:
            return [], 0
        places = self.index.get((history[current - 2] << 40) | (history[current - 1] << 20) | history[current], ())
        best, best_place, limit = 0, 0, self.max_match
        for place in places:
            if place < current:
                length = 0
                while length < limit and length <= place and history[place - length] == history[current - length]:
                    length += 1
                if length > best:
                    best, best_place = length, place
        if best < self.min_match:
            return [], 0
        tokens = history[best_place + 1:best_place + 1 + count]
        return (tokens * (count // len(tokens) + 1))[:count], best  # periodic text: the period goes on


class WindowCosts:
    """What a verification window of each size costs on this machine and what the drafts bring, measured as they run: the expected tokens per millisecond of a window of lookup drafts against the
    drafter's windows."""
    def __init__(self, max_rows, margin=0.1):
        self.max_rows, self.margin = max_rows, margin
        self.cost = [0.0] * (max_rows + 1)  # milliseconds, by rows
        self.seen = [0] * (max_rows + 1)
        self.good, self.bad = [0.0] * 4, [0.0] * 4  # lookup drafts held and windows cut, by match bucket
        self.drafter_tokens, self.drafter_rows = 1.5, 2.0  # of a drafter's window
        self.draft_ms = 0.0  # a step of the drafter's chain
        self.held, self.offered = [0.0] * CONFIDENCE_BUCKETS, [0.0] * CONFIDENCE_BUCKETS  # by the drafter's confidence
        self.expected = self.chance = 1.0  # tokens and chance of the chain being built
        self.passes = {}  # drafts of a drafter's window -> [passes, milliseconds of a whole pass: its draft chain and its verification]
        self.verdict_recorded = False  # the long chain's verdict of this machine went to chain_verdicts (once per runner)

    def cost_of(self, rows):
        if self.seen[rows]:
            return self.cost[rows]
        total = weight = 0.0
        for other in range(1, self.max_rows + 1):
            if self.seen[other]:
                share = min(self.seen[other], 20)
                total += share * self.cost[other] * (1 + SHAPE_PER_ROW * (rows - 1)) / (1 + SHAPE_PER_ROW * (other - 1))
                weight += share
        return total / weight if weight else 1 + SHAPE_PER_ROW * (rows - 1)

    @staticmethod
    def _bucket(match):
        return sum(match >= limit for limit in MATCH_BUCKETS)

    def rate(self, match):
        bucket = self._bucket(match)
        return (self.good[bucket] + PRIOR_WEIGHT * PRIOR_RATE[bucket]) / (self.good[bucket] + self.bad[bucket] + PRIOR_WEIGHT)

    def lookup_rows(self, available, match):
        """The rows of the window of lookup drafts that pays best, 0 if the drafter's window pays more."""
        rate, expected, chance, best, best_rows = self.rate(match), 1.0, 1.0, 0.0, 0
        for drafts in range(1, min(available, self.max_rows - 1) + 1):
            chance *= rate
            expected += chance
            value = expected / self.cost_of(drafts + 1)
            if value > best:
                best, best_rows = value, drafts + 1
        drafter = self.drafter_tokens / self.cost_of(min(self.max_rows, max(1, round(self.drafter_rows))))
        return best_rows if best > drafter * (1 + self.margin) else 0

    def _spent(self, rows):
        return self.cost_of(rows) + self.draft_ms * max(0, rows - 2)  # the first draft comes with the catch-up of the verified rows

    def start_chain(self):
        self.expected = self.chance = 1.0

    def keep_draft(self, rows, confidence):
        """Whether a draft of this confidence pays for the row it adds to a window of these rows: the tokens per millisecond of the window must grow, at the measured costs and the measured acceptance of this
        confidence (until measured, as likely as the drafter says)."""
        bucket = min(CONFIDENCE_BUCKETS - 1, int(confidence * CONFIDENCE_BUCKETS))
        chance = self.chance * (self.held[bucket] + PRIOR_WEIGHT * confidence) / (self.offered[bucket] + PRIOR_WEIGHT)
        if (self.expected + chance) / self._spent(rows + 1) <= self.expected / self._spent(rows):
            return False
        self.expected += chance
        self.chance = chance
        return True

    def observe_confidences(self, confidences, accepted):
        """Draft j was reached when the drafts before it held, it held when j < accepted."""
        for index, confidence in enumerate(confidences[:accepted + 1]):
            bucket = min(CONFIDENCE_BUCKETS - 1, int(confidence * CONFIDENCE_BUCKETS))
            self.held[bucket] = CONFIDENCE_FADE * self.held[bucket] + (index < accepted)
            self.offered[bucket] = CONFIDENCE_FADE * self.offered[bucket] + 1.0

    def observe_pass(self, drafts, milliseconds):
        entry = self.passes.setdefault(drafts, [0, milliseconds])
        entry[0] += 1
        entry[1] = min(entry[1], milliseconds) if entry[0] <= PASS_START else entry[1] + PASS_WEIGHT * (min(milliseconds, PASS_SPIKE * entry[1]) - entry[1])

    def pass_ratio(self, usual, long, samples):
        """The time of a pass with `long` drafts against one with `usual`, None until both were measured `samples` times."""
        short, longer = self.passes.get(usual), self.passes.get(long)
        return None if short is None or longer is None or min(short[0], longer[0]) < samples else longer[1] / short[1]

    def observe_draft(self, milliseconds):
        self.draft_ms += COST_WEIGHT * (milliseconds - self.draft_ms) if self.draft_ms else milliseconds

    def observe(self, rows, milliseconds, accepted, lookup, match=0):
        self.cost[rows] = (1 - COST_WEIGHT) * self.cost[rows] + COST_WEIGHT * milliseconds if self.seen[rows] else milliseconds
        self.seen[rows] += 1
        if lookup:
            bucket = self._bucket(match)
            self.good[bucket] = FADE * self.good[bucket] + accepted
            self.bad[bucket] = FADE * self.bad[bucket] + (accepted < rows - 1)
        else:
            self.drafter_tokens += TOKENS_WEIGHT * (accepted + 1 - self.drafter_tokens)
            self.drafter_rows += TOKENS_WEIGHT * (rows - self.drafter_rows)


MEASURED_FADE, MEASURED_PRIOR, MEASURED_WEIGHT, MEASURED_SAMPLES = 0.9, 0.6, 2.0, 3
MEASURED_IDLE = 0.97  # what is known of the acceptance, kept at this share per plain pass (no draft verified)
MEASURED_STEP = 0.15  # a draft more, against the largest count measured, while a single count with drafts is measured
MEASURED_PASS_WEIGHT, MEASURED_PASS_SPIKE = 0.1, 1.5  # a new pass in the mean time of its count; counted for at most this many times that mean (a capture, another program)
MEASURED_STALE = 32  # windows after which the time of a count next to the choice is measured again


class MeasuredDrafts:
    """The drafts of a window chosen from what this machine measures (an automatic count): the passes' mean times by number of drafts (0 drafts: a plain pass) and the recent
    acceptance of each draft position (draft k is verified when the drafts before it held). The count that gives the most tokens per millisecond is taken, 0 when no window pays for
    itself: drafts never decode slower than no drafts, whatever the context length, the cache layout, the weights' placement or the text. While no draft is verified (plain passes) what
    is known of the acceptance fades: the estimates return to their priors (the first position: MEASURED_PRIOR, the next ones: the rate of the position before) and drafts are tried
    again, so the choice never stays at 0 from a few unlucky windows. The count above the choice is measured once it can be chosen (one window in 4 until it has MEASURED_SAMPLES passes),
    and a count next to the choice that runs were measured with but no window in the last MEASURED_STALE is measured again (the count above in one window of MEASURED_STALE, the one
    below in another): a time raised by another program's work does not keep a count out, and a count never run (its graphs not captured yet) is only reached through the choice."""
    def __init__(self, costs, limit):
        self.costs, self.limit = costs, limit
        self.held, self.reached = [0.0] * limit, [0.0] * limit
        self.ms, self.samples = [0.0] * (limit + 1), [0] * (limit + 1)  # the mean time of a pass by its drafts (the first of each count left out: its graph is captured there)
        self.measured = [0] * (limit + 1)  # the window that last measured each count
        self.windows = 0

    def observe_pass(self, drafts, milliseconds):
        """Means, not the fastest passes: where experts run on the CPU the time of a pass varies with the experts its rows route to, and more rows reach more of them."""
        self.samples[drafts] += 1
        count = self.samples[drafts] - 1
        if count == 1:
            self.ms[drafts] = milliseconds
        elif count > 1:
            self.ms[drafts] += max(MEASURED_PASS_WEIGHT, 1.0 / count) * (min(milliseconds, MEASURED_PASS_SPIKE * self.ms[drafts]) - self.ms[drafts])
        self.measured[drafts] = self.windows
        if drafts == 0:
            for index in range(self.limit):
                self.held[index] *= MEASURED_IDLE
                self.reached[index] *= MEASURED_IDLE

    def observe(self, drafts, accepted):
        for index in range(min(drafts, accepted + 1)):
            self.held[index] = MEASURED_FADE * self.held[index] + (index < accepted)
            self.reached[index] = MEASURED_FADE * self.reached[index] + 1.0

    def pass_times(self, limit):
        """Milliseconds of a pass by drafts, 0 to limit: measured (a count's second pass on), else a line between the measured counts around it; above the largest, the step of the two
        largest measured counts with drafts (one alone: MEASURED_STEP of its time); below the smallest, SHAPE_PER_ROW per draft. Nothing measured yet: relative units."""
        known = [count for count in range(self.limit + 1) if self.samples[count] > 1]
        if not known:
            return [1 + SHAPE_PER_ROW * drafts for drafts in range(limit + 1)]
        drafted, times = [count for count in known if count], []
        for drafts in range(limit + 1):
            below, above = max((count for count in known if count <= drafts), default=None), min((count for count in known if count >= drafts), default=None)
            if below == drafts:
                times.append(self.ms[drafts])
            elif below is not None and above is not None:
                times.append(self.ms[below] + (self.ms[above] - self.ms[below]) * (drafts - below) / (above - below))
            elif below is None:
                times.append(self.ms[above] * (1 + SHAPE_PER_ROW * drafts) / (1 + SHAPE_PER_ROW * above))
            else:
                step = (self.ms[drafted[-1]] - self.ms[drafted[-2]]) / (drafted[-1] - drafted[-2]) if len(drafted) > 1 else MEASURED_STEP * self.ms[below]
                times.append(self.ms[0] * (1 + SHAPE_PER_ROW) + step * (drafts - 1) if below == 0 else self.ms[below] + step * (drafts - below))
        return times

    def best(self, limit):
        times = self.pass_times(limit)
        rate, expected, chance, best, best_value = MEASURED_PRIOR, 1.0, 1.0, 0, 1.0 / times[0]
        for drafts in range(1, limit + 1):
            rate = (self.held[drafts - 1] + MEASURED_WEIGHT * rate) / (self.reached[drafts - 1] + MEASURED_WEIGHT)  # little known of a position lately: as the one before it
            chance *= rate
            expected += chance
            value = expected / times[drafts]
            if value > best_value:
                best, best_value = drafts, value
        return best

    def choose(self, limit):
        limit = min(limit, self.limit)
        self.windows += 1
        best = self.best(limit)
        if best < limit and self.samples[best + 1] <= MEASURED_SAMPLES and self.windows % 4 == 0:
            return best + 1  # the count above the choice is not measured yet
        phase = self.windows % MEASURED_STALE
        if phase == 0 and best < limit and self.samples[best + 1] and self.windows - self.measured[best + 1] >= MEASURED_STALE:
            return best + 1
        if phase == MEASURED_STALE // 2 and best > 0 and self.windows - self.measured[best - 1] >= MEASURED_STALE:
            return best - 1
        return best


CONFIDENT_EDGES = (0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.97, 0.99)  # bins of a draft's probability under its drafter (the drafted token's)
CONFIDENT_PRIOR = {False: ((0.135, 0.263, 0.424, 0.506, 0.686, 0.752, 0.87, 0.872, 0.981), (0.039, 0.193, 0.299, 0.401, 0.557, 0.659, 0.715, 0.842, 0.959)),
                   True: ((0.311, 0.38, 0.367, 0.417, 0.58, 0.509, 0.576, 0.73, 0.94), (0.223, 0.3, 0.31, 0.383, 0.328, 0.391, 0.52, 0.596, 0.93))}  # rate of a draft reached, by bin: the first draft, the next ones; greedy, sampled (Inkling 2's DSpark head on prose, tool JSON and code, 6K and 28K contexts)
CONFIDENT_WEIGHT, CONFIDENT_FADE = 4.0, 0.98  # the prior counts for this many windows of its bin; a bin's measures fade at each new one
CONFIDENT_CAPTURE_MS = 250.0  # the first pass of a count: its graphs are captured there (218 to 344 ms at a 28K context on an RTX 5090, against 17 to 24 after)
CONFIDENT_UNIT_MS = 15.0  # a plain pass, while no pass is measured yet (pass_times in relative units)
CONFIDENT_SHORT, CONFIDENT_RECENT = 3, 8  # the short chain, while none of the last CONFIDENT_RECENT windows took all its drafts


class ConfidentDrafts:
    """The drafts of each window chosen from the drafter's own probabilities of the drafts of that window: each draft's probability, calibrated on the running text (the rate
    at which drafts of that probability bin held, for the first draft and the next ones, from a prior fitted offline), gives the expected tokens of each count, and the count that
    gives the most tokens per millisecond at the pass times measured by `measured` (a MeasuredDrafts, also kept as the history-only rule) is taken. A count never run (its graphs
    not captured) is taken only once the tokens it would have gained since add up to what its capture costs; until then the best of the counts already run."""
    def __init__(self, measured):
        self.measured = measured
        bins = len(CONFIDENT_EDGES) + 1
        self.held = {sampled: [[0.0] * bins, [0.0] * bins] for sampled in (False, True)}
        self.reached = {sampled: [[0.0] * bins, [0.0] * bins] for sampled in (False, True)}
        self.credit = [0.0] * (measured.limit + 1)
        self.recent = []  # the counts of the latest CONFIDENT_RECENT windows

    def span(self, limit, sampled):
        """The drafts of the chain to compute for the next window: its count is chosen among them. A sampled draft costs about 0.45 ms of the
        chain: CONFIDENT_SHORT while the recent windows took fewer, the whole chain once one took them all; a greedy draft costs little, the
        whole chain always (shortened, greedy code and prose lost 3 to 9 % at 56K). Two lengths only: a chain of another length is another
        graph, captured at its first use (a chain following the recent choices paid 30-60 ms captures)."""
        return limit if not sampled or max(self.recent, default=limit) >= CONFIDENT_SHORT else min(limit, CONFIDENT_SHORT)

    @staticmethod
    def _bin(probability):
        return sum(probability >= edge for edge in CONFIDENT_EDGES)

    def rates(self, confidences, sampled):
        rates = []
        for index, probability in enumerate(confidences):
            group, cell = min(index, 1), self._bin(probability)
            held, reached = self.held[sampled][group][cell], self.reached[sampled][group][cell]
            rates.append((held + CONFIDENT_WEIGHT * CONFIDENT_PRIOR[sampled][group][cell]) / (reached + CONFIDENT_WEIGHT))
        return rates

    def choose(self, confidences, limit, sampled):
        """The drafts to verify among the first `limit` of a chain whose drafts have these probabilities."""
        measured = self.measured
        limit = min(limit, len(confidences), measured.limit)
        times, expected, chance = measured.pass_times(limit), [1.0], 1.0
        for rate in self.rates(confidences[:limit], sampled):
            chance *= rate
            expected.append(expected[-1] + chance)
        values = [expected[drafts] / times[drafts] for drafts in range(limit + 1)]
        best = max((drafts for drafts in range(limit + 1) if drafts == 0 or measured.samples[drafts]), key=lambda drafts: values[drafts])
        unit = 1.0 if any(count > 1 for count in measured.samples) else CONFIDENT_UNIT_MS  # milliseconds of a unit of `times`
        unlocked = None
        for drafts in range(limit + 1):  # the counts never run that would do better: what each would bring, against its capture
            if values[drafts] > values[best] and not measured.samples[drafts]:
                self.credit[drafts] += (values[drafts] - values[best]) * times[best]
                if self.credit[drafts] >= CONFIDENT_CAPTURE_MS / unit * values[best] and (unlocked is None or values[drafts] > values[unlocked]):  # the tokens a capture costs
                    unlocked = drafts
        best = best if unlocked is None else unlocked
        self.recent = [*self.recent[1 - CONFIDENT_RECENT:], best]
        return best

    def observe(self, confidences, drafts, accepted, sampled):
        """Draft k was reached when the drafts before it held, it held when k < accepted."""
        for index in range(min(drafts, accepted + 1)):
            group, cell = min(index, 1), self._bin(confidences[index])
            self.held[sampled][group][cell] = CONFIDENT_FADE * self.held[sampled][group][cell] + (index < accepted)
            self.reached[sampled][group][cell] = CONFIDENT_FADE * self.reached[sampled][group][cell] + 1.0
