---
name: long-story-writing
description: Plan, draft and revise stories longer than one context window (novels, serials, long scripts) in files, keeping chronology, character knowledge, evidence, causality and promises consistent through a canon index and a final consistency gate.
---

# Long Story Writing

Keep the authoritative manuscript in `@documents/story.md`, the scene plan in `@documents/outline.md`, and a compact canon index in `@documents/continuity.md`. Use `rg` for bounded searches, `append_text` for new units and `edit` for exact revisions; do not rebuild the story, its plan or its canon from conversational memory or a compaction summary.

For a story the reader plays through (choices, clues, puzzles, branches), plan its steps with wangp://skills/progression-design as well.

Before drafting, write the outline: premise, genre, voice, point of view and tense, setting rules, then one entry per chapter or scene with its purpose, the characters present, what changes and the promises it opens or pays off. Give every unit a unique, searchable heading such as `## Chapter 03 - The Flooded Archive`, identical in the outline and the manuscript.

The canon index holds one fact per line, starting with a type and a stable ID so that one search finds everything known about a name: `PERSON:`, `PLACE:`, `OBJECT:`, `RULE:`, `KNOWS:` (who knows what, since which unit), `EVENT:` (unit, what happened, consequence) and `PROMISE:` (where it was opened, status, planned payoff), for example `EVENT: ch05 | the archive floods | the ledger is lost; Ilan blames Mara`. Keep stable descriptions apart from current state: update a changed state by editing its line, and keep the history in `EVENT:` lines.

Track chronology (day, time of day, time elapsed between units), character knowledge, evidence and clues, causal dependencies (nothing may rely on an event that has not happened yet) and unresolved promises. A fact the story will rely on later belongs in the index as soon as it is written.

Reuse facts that are still in recent context, group related searches into one call, and read each action's contract once rather than before every call.

Draft one unit at a time:

1. Usually make one preflight `rg` call for earlier facts used by this unit. Combine related terms with repeated `-e`, for example `-n -F -e "Mara" -e "lantern-key" -e "ch04" -- @documents/continuity.md`, and search `@documents/story.md` with `-C 3` only when the exact earlier wording matters.
2. Write the unit as literal prose and add it with `append_text(file_path, text)`: start it with its heading, separate it from the previous unit with one blank line, include every newline, and never paste earlier units back into the call.
3. While the new unit remains in recent context, check it against the canon facts and promises it uses. When the append result includes `structure`, confirm that `last_heading` is the expected heading and that `headings_added` and `last_heading_occurrences` are 1; investigate any mismatch before continuing.
4. Append the unit's new facts, events and promises to the canon index, and `edit` the lines whose state changed; mark a payoff by editing its promise's status.
5. Mark the unit as drafted in the outline, and record any change of plan there before drafting the next unit.
6. Keep each unit near the length its outline entry gives it; when a unit outgrows its plan, split it into two planned units in the outline rather than writing past it.

After a context compaction the files are authoritative: find the last heading in the manuscript (`-n "^## " -- @documents/story.md`), read the outline's next entry and search the index for the facts it needs. Never redraft a unit from a summary.

Revise with `edit(file_path, old_string, new_string)`: copy an exact, unique old string (include a heading or a neighboring sentence when needed), keep whitespace and line endings, then search the affected names again. For a global change such as a renamed character or a moved date, record the decision in the outline first, list every occurrence with `-n -F`, then edit each passage; use `replace_all=true` only for a verified literal rename.

For structural review, search headings and outline entries instead of reading the whole manuscript; read bounded passages (`-C` around a match) for local prose review.

## Final consistency gate

Before delivering, verify the whole story from the files, not from memory:

1. List the manuscript headings and compare their order and count with the outline: every planned unit is present exactly once.
2. Check every `PROMISE:` line: each is paid off, or deliberately left open with the reason noted in the outline.
3. Check chronology: day and time markers never go backwards unless the outline calls for a flashback, and travel, healing and waiting times are plausible.
4. Check knowledge and causality: no character acts on information before the `KNOWS:` or `EVENT:` line that gives it, and every consequence follows its cause.
5. Inspect the bounded source passages whenever search results or the canon index cannot prove consistency. If the index is incomplete, contradictions appear, or a fact was never recorded, read the passages that establish it, fix the story or the index with exact edits, and repeat the affected checks.
6. Search for names spelled several ways and for leftover placeholders.
7. Read the opening and the ending as bounded passages: the ending answers the premise, and names, ages and places match the canon.

Report which checks ran, what they found and what was fixed, and name any thread deliberately left open.

When delivering the manuscript, return its authorized `@documents` path. Do not paste the whole story into chat unless the user explicitly requests it.
