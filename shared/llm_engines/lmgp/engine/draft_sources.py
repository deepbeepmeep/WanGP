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
