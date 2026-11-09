# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Draft trees for tree verification.

A tree has a static shape: each node is a path of candidate ranks from the anchor, (0,) the drafter's best next token, (0, 1) the second best
after it, and so on. Parents, depths, the attention bits and the paths the linear-attention layers run are fixed: one CUDA graph per tree,
no host read. Only the tokens change at each pass, read from a drafter lattice: for draft row i, scores of K candidates given the lane
chosen on row i-1 (DSpine's refinement table; row 0's sender is the anchor).

Verification samples the target's token at each visited node, under all its rules, and follows the child that drafted it: the output is
the target's own sampling whatever the tree, and a node is reached with the target's probability of its path.
"""
from __future__ import annotations

import torch

# Best-first over the rank statistics of DSpine on Deepy's prose (the target's next token is the drafter's top 1 in 53 % of the cases, its
# top 4 in 80 %), each level 0.85 as likely as the one above: 15 drafts, 16 verification rows, under the cost step past 16 rows.
DEFAULT_TREE = ((0,), (1,), (2,), (3,), (4,), (0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, 0, 0))


class StaticTree:
    """Node 0 is the anchor, nodes 1.. the drafts in breadth order (parents before children)."""

    def __init__(self, rank_paths=DEFAULT_TREE):
        paths = sorted({tuple(path) for path in rank_paths}, key=lambda path: (len(path), path))
        if any(path[:-1] and path[:-1] not in paths for path in paths):
            raise ValueError("A draft tree needs every node's parent.")
        self.rank_paths = paths
        index = {(): 0, **{path: node + 1 for node, path in enumerate(paths)}}
        self.size = len(paths) + 1
        self.depth = max(len(path) for path in paths)
        self.parents = [-1] + [index[path[:-1]] for path in paths]
        self.depths = [0] + [len(path) for path in paths]
        self.ranks = [0] + [path[-1] for path in paths]
        lineage = [[0]]
        for node in range(1, self.size):
            lineage.append(lineage[self.parents[node]] + [node])
        self.lineage = lineage  # the nodes from the anchor to each node
        self.bits = [sum(1 << ancestor for ancestor in line) for line in lineage]
        leaves = [node for node in range(1, self.size) if node not in self.parents]
        self.paths = [lineage[leaf] + [leaf] * (self.depth + 1 - len(lineage[leaf])) for leaf in leaves]  # padded by repeating the leaf
        rows = {}
        for number, path in enumerate(self.paths):
            for step, node in enumerate(lineage[path[-1]]):
                rows.setdefault(node, number * (self.depth + 1) + step)
        self.rows = [rows[node] for node in range(self.size)]  # where each node is computed among the paths' rows

    def tensors(self, device):
        """parents, depths, ancestor bits (int64), paths [P, depth + 1], rows [N]: what the verification pass reads."""
        long = dict(dtype=torch.long, device=device)
        return (torch.tensor(self.parents, **long), torch.tensor(self.depths, **long), torch.tensor(self.bits, **long),
                torch.tensor(self.paths, **long), torch.tensor(self.rows, **long))

    def draft(self, table: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """The tokens of the tree [N] (node 0: 0, the anchor goes there) from a lattice: table [n, K, K] (row, sender lane, candidate)
        and candidates [n, K]. Fixed shapes, no host read."""
        order = table.argsort(dim=-1, descending=True)  # [n, K, K]: candidate lanes by rank, per sender lane
        lanes = [torch.zeros((), dtype=torch.long, device=table.device)]
        tokens = [torch.zeros((), dtype=torch.long, device=table.device)]
        for node in range(1, self.size):
            row, parent = self.depths[node] - 1, self.parents[node]
            sender = lanes[parent] if parent else lanes[0]  # row 0: every sender lane is the anchor
            lane = order[row].index_select(0, sender.reshape(1))[0, self.ranks[node]]
            lanes.append(lane)
            tokens.append(candidates[row].gather(0, lane.reshape(1))[0])
        return torch.stack(tokens)


def apply_tree_rules(logits, tokens, ancestors, depths, history, bias, suppressed, thinking_stops, thinking, penalty, exempt=None):
    """dflash_sampling.apply_rules for a tree: row r continues the path anchor -> r, so its virtual tokens are r's ancestors' (and its
    own) drafted tokens, not the rows before it. ancestors [N, N] bool."""
    rows, vocab = logits.shape
    scores = logits.clone()
    drafted = ancestors.clone()
    drafted[:, 0] = False  # the anchor is already in the history
    if penalty != 1:
        # Branches share tokens: count each row's drafted ancestors per token (sums of 0/1, exact in any order) instead of a boolean scatter.
        counts = torch.zeros((rows, vocab), dtype=torch.float32, device=logits.device).scatter_add_(1, tokens[None, :].expand(rows, -1), drafted.float())
        seen = history.unsqueeze(0) | (counts > 0)
        if exempt is not None:
            seen &= ~exempt
        scores = torch.where(seen, torch.where(scores < 0, scores * penalty, scores / penalty), scores)
    scores.add_(bias)
    scores.masked_fill_(suppressed.unsqueeze(0), -torch.inf)
    close, remaining, active = thinking.unbind()
    closed = (drafted & (tokens[None, :] == close)).any(dim=1)
    in_thinking = (active != 0) & ~closed
    force = in_thinking & (depths >= remaining)
    scores.masked_fill_(in_thinking[:, None] & thinking_stops[None, :], -torch.inf)
    scores = torch.where(force[:, None], torch.where(torch.arange(vocab, device=logits.device)[None, :] == close, 0., -torch.inf).to(scores.dtype), scores)
    return scores


def accept_tree(probabilities, unsafe, tokens, parents, noise, stops, max_depth, greedy=False):
    """Walk the tree from the anchor: each visited node's token is sampled from its target row (probabilities[r], under all rules; the
    argmax when greedy) and the walk moves to the child that drafted it. Returns [count, accepted, fallback, path nodes (max_depth),
    emitted tokens (max_depth + 1)], int64, -1 past the end: the emitted tokens are the accepted drafts then the last sampled token, cut
    after a stop token. fallback: an unsafe row (ambiguous nucleus support) is in the tree, decided before any random draw is used."""
    sampled = probabilities.argmax(dim=-1) if greedy else (probabilities / noise).argmax(dim=-1)  # [N]: one draw per row
    node = torch.zeros((), dtype=torch.long, device=tokens.device)
    alive = torch.ones((), dtype=torch.bool, device=tokens.device)
    accepted = torch.zeros((), dtype=torch.long, device=tokens.device)
    path, emitted = [], []
    for _ in range(max_depth + 1):
        token = sampled.gather(0, node.reshape(1))[0]
        emitted.append(torch.where(alive, token, -1))
        match = (parents == node) & (tokens == token)
        follow = alive & match.any() & ~stops.gather(0, token.reshape(1))[0]
        child = match.long().argmax()
        path.append(torch.where(follow, child, -1))
        alive = alive & follow
        accepted = accepted + follow.long()
        node = torch.where(follow, child, node)
    count = (torch.stack(emitted) >= 0).long().sum()
    fallback = unsafe.any().long()  # any row of the tree, decided before the draws
    return torch.cat((torch.stack((count, accepted, fallback)), torch.stack(path[:-1]), torch.stack(emitted)))


def walk_reference(parents, tokens, sample_at):
    """The visited path: sample_at(node) gives the target's token at a node (its sampled next token); the walk follows the child that
    drafted it. Returns (accepted nodes in order, the final sampled token)."""
    children = {}
    for node in range(1, len(parents)):
        children.setdefault(int(parents[node]), {})[int(tokens[node])] = node
    node, path = 0, []
    while True:
        token = sample_at(node)
        child = children.get(node, {}).get(token)
        if child is None:
            return path, token
        path.append(child)
        node = child
