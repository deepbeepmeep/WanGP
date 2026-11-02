---
name: progression-design
description: "Use before making anything the user goes through in several steps: game levels, puzzles, quests, an adventure, an interactive story or mystery, an escape room, a course or quiz in stages. Plan the whole progression ahead as data, prove with a search that every step is possible, needed and as long as intended, then build it and play it through."
---

# Progression design

Something that is over after one action is a demo. Length and interest come from steps that build on each other: what the player finds, learns, masters or decides now matters several steps later. Plan the whole progression before making any asset or writing any code, in `@documents/<name>-plan.md`; then keep the content as data that a search checks.

## 1. Plan the progression

1. **Goal**: what the player wants, what stands in the way, how it ends (and the other endings, if any). Decide hidden facts first (the culprit, the code, where the exit is): every clue, obstacle and level follows from them.
2. **Size**: what the user asked for, otherwise about 8 to 12 meaningful steps for a short piece (actions that change something; moving around is free), or 5 to 8 levels.
3. **Work backwards** from the end: what does the last step need? Where is that obtained, and what does that need? One line per step: `S07 open the safe | needs: code (S04), study unlocked (S05) | gives: the will`. Levels the same way: `L4 | new: moving platforms | uses: double jump (L2), keys (L3)`.
4. **Every step required, and chained**: a step the player can skip does not count, and clues all open from the start with an ending that needs them all make a checklist, not a progression. Make most findings needed by a later step: searching the coal bin needs the footprints that point to it, the accusation needs the motive as well as the proof. A clue that is only flavour is a red herring, and one or two are enough.
5. **Shape**:
   - Several threads open at once that join later, so the player chooses an order and always has something to try.
   - Each level or stage introduces one new idea, lets the player practise it, then combines it with earlier ones; difficulty rises with a breather after a peak.
   - At least one step that needs two earlier findings or skills together; a choice whose consequence appears several steps later.
   - Endings that close the game (an accusation, a final choice, the exit) open once the player has what it takes to decide, not from the first action.
6. **Feedback**: what the player sees or reads after each action, and why a refused action does not work yet ("The drawer is locked. Someone keeps the key close."), so the next step can be guessed; a hint after a few failures.

For the prose of a long story, keep the outline and the canon facts as the long-story-writing skill does, and the steps in the data below.

## 2. The content as data

Read by a generic engine, so changing the design never means changing code.

- **Steps, quests, stories, item puzzles**: `apps/<name>/assets/story.json` with `start` (the initial facts), `steps` (the planned number of actions to the first ending), `actions` and `endings`. A fact is a string: `has:key`, `knows:alibi-false`, `door-open`, `trust:butler`, and `at:study` for where the player is. Each action: `id`, `label`, its place or character, `requires`, `gives`, `removes`, its `text`, and `ending` when it ends the game (an accusation); an ending can instead have `requires`. An action that only changes `at:` facts is a free move.
- **Levels**: grids of characters (`#` wall or ground, `.` empty, `S` start, `G` goal, `k`/`D` key and door, `^` hazard...), one list of rows per level, with the rules the search needs (jump height and width in tiles).

## 3. Prove it before writing the app

`print(wangp.app.progression('apps/<name>/assets/story.json'))` searches every order of play: the shortest path to each ending (moves not counted), endings never reached, dead ends, states that can no longer reach the main ending, facts given but never needed (red herrings, and steps that can be skipped) and actions never possible. `app_check` reports the same search whenever the app has `assets/story.json`. Fix the data until the first ending takes the planned number of steps and nothing else is flagged except the red herrings.

Grid levels: a search over positions (and keys held) with the game's own moves; the shortest path should grow from level to level. Puzzles whose state holds several movable things (boxes, blocks, ice) grow fast: keep levels small (about 8 x 8, at most four boxes), drop states with a box stuck in a corner that is not a target, search one level per run, and pass `timeout_s` (up to 600) to a run that searches several.

```python
import json
from collections import deque

def solve(rows):
    cells = {(x, y): c for y, row in enumerate(rows) for x, c in enumerate(row)}
    start = next(p for p, c in cells.items() if c == 'S')
    seen, queue = {(start, frozenset())}, deque([(start, frozenset(), 0)])
    while queue:
        (x, y), keys, steps = queue.popleft()
        if cells[(x, y)] == 'G':
            return steps
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # a platformer: the standing cells reachable by a walk, a fall or a jump instead
            p = (x + dx, y + dy)
            c = cells.get(p, '#')
            if c in '#^' or (c == 'D' and 'k' not in keys):
                continue
            state = (p, keys | ({'k'} if c == 'k' else set()))
            if state not in seen:
                seen.add(state)
                queue.append((*state, steps + 1))
    return None
print([solve(level) for level in json.load(open('apps/maze/assets/levels.json'))])  # None: impossible
```

## 4. Build and play it through

- Assets follow the plan: one picture per place or level theme, characters, items and tiles (transparent), voices, sound effects (models with the `sound effects` speciality), music, made in batches.
- Clickable things: DOM elements with stable ids (`#act-open-safe`, `#item-key`) that `app_check` clicks by selector, or regions of a canvas whose coordinates the plan records. `window.wangpState()` returns the facts or level, the step count and the ending. A level select (`#level-3` buttons, or keys 1 to 9) lets checks start at any level.
- Play the walkthrough with `app_check` (a `shot` at the key moments): the state shows each step happen and the ending only at the last one. Then try one action too early and read its refusal.
- Report the number of steps or levels, the endings and how each is reached.
