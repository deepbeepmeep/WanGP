# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The progression of a story, quest or puzzle written as data (facts, actions and endings): every order of play is searched, so a design is
known to be finishable, as long as planned and free of dead ends before anyone plays it. app_check reports it for assets/story.json."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

PLACE_PREFIXES = ("at:", "in:", "in-")  # facts naming where the player is: an action that only changes them is a free move
MAX_STATES = 100_000
_FOUND: dict[tuple, list] = {}  # findings per story file version: app_check runs often, the search can take seconds


def _entries(value: Any) -> list[dict[str, Any]]:
    """A list of objects, or an object of them keyed by id."""
    if isinstance(value, dict):
        return [{"id": name, **entry} for name, entry in value.items() if isinstance(entry, dict)]
    return [entry for entry in value or [] if isinstance(entry, dict)]


def _facts(value: Any) -> frozenset[str]:
    return frozenset([value] if isinstance(value, str) else value or [])


def check(story: dict[str, Any]) -> dict[str, Any] | None:
    """The search of a story's every order of play, or None when it holds no actions with requires or gives. Moves between places are not counted."""
    actions = _entries(story.get("actions"))
    if not any("requires" in action or "gives" in action for action in actions):
        return None
    endings = _entries(story.get("endings"))
    ended_by = {str(action["ending"]) for action in actions if action.get("ending")}
    requires = {str(entry["id"]): _facts(entry.get("requires")) for entry in endings}  # an ending an action leads to: its requires are that action's too
    conditions = {name: needs for name, needs in requires.items() if needs and name not in ended_by}  # the others end the game once their facts are held
    rules = [(str(action.get("id") or action.get("label") or index), _facts(action.get("requires")) | requires.get(str(action.get("ending")), frozenset()), _facts(action.get("gives")),
              _facts(action.get("removes")), action.get("ending")) for index, action in enumerate(actions)]
    ending_of = lambda state: next((name for name, needs in conditions.items() if needs <= state), None)
    start = _facts(story.get("start"))
    best, previous, queue = {start: 0}, {start: None}, deque([start])
    edges, reached, possible = {}, {}, set()

    def path(state):
        names = []
        while previous[state] is not None:
            state, name, cost = previous[state]
            names += [name] if cost else []
        return names[::-1]

    def reach(ending, steps, names):
        if ending not in reached or steps < reached[ending][0]:
            reached[ending] = (steps, names)

    while queue and len(best) <= MAX_STATES:
        state = queue.popleft()
        if (ending := ending_of(state)) is not None:
            reach(ending, best[state], path(state))
            edges[state] = [(None, ending)]  # an ending: nothing follows
            continue
        edges[state] = []
        for name, needs, gives, removes, ending in rules:
            if not needs <= state:
                continue
            possible.add(name)  # text-only actions too (a refusal, a description)
            if ending:
                edges[state].append((None, str(ending)))
                reach(str(ending), best[state] + 1, path(state) + [name])
                continue
            following = (state | gives) - removes
            if following == state:
                continue
            edges[state].append((following, None))
            cost = 0 if all(fact.startswith(PLACE_PREFIXES) for fact in gives | removes) else 1
            if following not in best or best[state] + cost < best[following]:
                best[following], previous[following] = best[state] + cost, (state, name, cost)
                (queue.appendleft if cost == 0 else queue.append)(following)
    def able(target):  # the states from which an ending (any ending when target is None) can still be reached
        alive = {state for state, nexts in edges.items() if any(following is None and ending == (target or ending) for following, ending in nexts)}
        while grown := {state for state, nexts in edges.items() if state not in alive and any(following in alive for following, _ in nexts if following is not None)}:
            alive |= grown
        return alive

    named = [str(entry["id"]) for entry in endings] or list(reached)
    any_ending = able(None)
    dead = sorted((state for state in edges if state not in any_ending), key=lambda state: best[state])
    main_ending = able(named[0]) if named else any_ending
    lost = sorted((state for state in any_ending if state not in main_ending and ending_of(state) is None), key=lambda state: best[state])
    needed = set().union(*(needs for _, needs, *_ in rules), *conditions.values())
    around, frontier = {start}, [start]  # the states reached by moves alone: what the player can do from the start, in any order
    while frontier:
        state = frontier.pop()
        for _, needs, gives, removes, ending in rules:
            if needs <= state and not ending and gives | removes and all(fact.startswith(PLACE_PREFIXES) for fact in gives | removes):
                if (following := (state | gives) - removes) not in around:
                    around.add(following)
                    frontier.append(following)
    open_names = {name for name, needs, *_ in rules if any(needs <= state for state in around)}
    main_path = reached[named[0]][1] if named and named[0] in reached else []
    return {
        "states": len(best), "complete": not queue,
        "shortest": {ending: {"actions": steps, "path": names} for ending, (steps, names) in sorted(reached.items(), key=lambda item: item[1][0])},
        "never_reached": [ending for ending in named if ending not in reached],
        "dead_ends": len(dead), "dead_end_example": path(dead[0]) if dead else None,
        "main_lost": len(lost), "main_lost_example": path(lost[0]) if lost else None,
        "never_needed": sorted(fact for _, _, gives, *_ in rules for fact in gives if fact not in needed and not fact.startswith(PLACE_PREFIXES)),
        "never_possible": [name for name, *_ in rules if name not in possible],
        "planned": story.get("steps"), "main_ending": named[0] if named else None, "open_at_start": sum(name in open_names for name in main_path),
    }


def findings(file: Path) -> list[dict[str, Any]]:
    """app_check findings for a story file: the search's summary, and warnings for what a player would meet as a flaw."""
    name, stat = f"assets/{file.name}", file.stat()
    key = (str(file), stat.st_mtime_ns, stat.st_size)
    if key not in _FOUND:
        _FOUND.clear()
        _FOUND[key] = _findings(file, name)
    return _FOUND[key]


def _findings(file: Path, name: str) -> list[dict[str, Any]]:
    try:
        result = check(json.loads(file.read_text(encoding="utf-8")))
    except (ValueError, AttributeError, TypeError) as error:
        return [{"level": "error", "file": name, "message": f"{name} could not be searched: {error}"}]
    if result is None:
        return []
    shortest = "; ".join(f"{ending} in {item['actions']} actions ({' > '.join(item['path'])})" for ending, item in result["shortest"].items()) or "no ending reached"
    found = [{"level": "info", "file": name, "message": f"Progression searched over {result['states']} states (moves between places not counted): {shortest}."
              + (f" Given but never needed (fine for red herrings): {', '.join(result['never_needed'])}." if result["never_needed"] else "")}]
    warn = lambda message: found.append({"level": "warning", "file": name, "message": message})
    main = result["shortest"].get(result["main_ending"])
    if result["planned"] and main and main["actions"] < int(result["planned"]):
        warn(f"The {result['main_ending']} ending takes {main['actions']} actions, the plan (steps) says {result['planned']}: make the steps the player can skip require what earlier ones give, "
             "or set steps to the number of actions the player must do.")
    if main and main["actions"] >= 4 and 3 * result["open_at_start"] > 2 * main["actions"]:
        warn(f"{result['open_at_start']} of the {main['actions']} actions to the {result['main_ending']} ending are possible from the start, in any order: a checklist more than a progression. "
             "Make some of them need what an earlier one gives (a clue pointing to where the next one is, a key, a testimony).")
    if result["never_reached"]:
        warn(f"Endings never reached: {', '.join(result['never_reached'])}.")
    if result["dead_ends"]:
        warn(f"{result['dead_ends']} states lead to no ending, for example after {' > '.join(result['dead_end_example']) or 'the start'}.")
    if result["main_lost"]:
        warn(f"{result['main_lost']} states can no longer reach the {result['main_ending']} ending, for example after {' > '.join(result['main_lost_example'])}: fine only when that is meant.")
    if result["never_possible"]:
        warn(f"Actions never possible: {', '.join(result['never_possible'])}.")
    if not result["complete"]:
        warn(f"The search stopped at {MAX_STATES} states: the results cover only part of the game.")
    return found
