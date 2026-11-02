# Mesh to flexible dual grid, PyTorch port of o-voxel's mesh_to_flexible_dual_grid (github.com/microsoft/TRELLIS.2/o-voxel,
# src/convert/flexible_dual_grid.cpp, MIT License, see LICENSE_TRELLIS2 in this folder), vectorized on the mesh's device.
# The active voxels are the four voxels around every crossing of a grid edge line with a triangle (the upstream scanline, here an
# enumeration of the lattice lines in each triangle's bounding box with the same half-open edge rules); each voxel's dual vertex
# minimizes the quadric of the planes of its crossings, of the triangles overlapping it, of the open boundary edges through it and a
# pull towards the mean of its crossings, constrained to the voxel (faces, edges and corners enumerated as upstream).
# Positions are handled in voxel units, which leaves the solution unchanged (the quadrics are scale equivariant).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
import torch.nn.functional as F

_TRIANGLES = 1 << 18  # triangles per chunk of the scanline and overlap enumerations
_SOLVE_ROWS = 1 << 20


def _plane_quadrics(triangles):
    """Quadric p p^T of each triangle plane p = (n, -n.v0) (F, 10): xx xy xz xw yy yz yw zz zw ww. Degenerate triangles get zeros."""
    v0, v1, v2 = triangles.unbind(1)
    n = F.normalize(torch.linalg.cross(v1 - v0, v2 - v1), dim=-1)
    p = torch.cat([n, -(n * v0).sum(-1, keepdim=True)], dim=-1)
    i, j = torch.triu_indices(4, 4, device=triangles.device)
    return p[:, i] * p[:, j]


def _enumerate(lo, hi):
    """Ranges [lo, hi] per row (R, D) enumerated: (row index (M,), offsets (M, D)) of every lattice point of each box."""
    sizes = (hi - lo + 1).clamp_min(0)
    counts = sizes.prod(-1)
    rows = torch.repeat_interleave(torch.arange(len(counts), device=lo.device), counts)
    local = torch.arange(len(rows), device=lo.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    offsets = []
    for axis in reversed(range(lo.shape[1])):
        size = sizes[rows, axis]
        offsets.append(lo[rows, axis] + local % size)
        local = local // size
    return rows, torch.stack(offsets[::-1], dim=1)


def _crossings(triangles, grid):
    """Crossings of the grid edge lines with the triangles (voxel units): for each axis a, the lines parallel to a at integer
    coordinates (k0, k1) of the two other axes, 1 <= k < grid. Returns (triangle index, axis, crossing point (3,)) of each crossing."""
    tri_ids, axes, points = [], [], []
    for ax2 in range(3):
        ax0, ax1 = (ax2 + 1) % 3, (ax2 + 2) % 3
        t = triangles[..., [ax0, ax1, ax2]].double()
        a, b, c = t.unbind(1)
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        flip = area < 0
        b, c = torch.where(flip[:, None], c, b), torch.where(flip[:, None], b, c)
        keep = (area != 0).nonzero().squeeze(1)
        lo = (t[keep, :, :2].amin(1).trunc() + 1).long().clamp(1, grid - 1)
        hi = t[keep, :, :2].amax(1).trunc().long().clamp(max=grid - 1)
        rows, lattice = _enumerate(lo, hi)
        tri = keep[rows]
        p = lattice.double()
        a, b, c = a[tri], b[tri], c[tri]
        inside = torch.ones(len(tri), dtype=torch.bool, device=p.device)
        weights = []
        for start, end in ((a, b), (b, c), (c, a)):  # counterclockwise: the interior is left of each edge
            dx, dy = end[:, 0] - start[:, 0], end[:, 1] - start[:, 1]
            e = dx * (p[:, 1] - start[:, 1]) - dy * (p[:, 0] - start[:, 0])
            inclusive = (dy > 0) | ((dy == 0) & (dx < 0))  # upstream scanline: right and top edges belong to the triangle
            inside &= (e > 0) | ((e == 0) & inclusive)
            weights.append(e)
        tri, p = tri[inside], p[inside]
        e_ab, e_bc, e_ca = (w[inside] for w in weights)
        z = (e_bc * a[inside][:, 2] + e_ca * b[inside][:, 2] + e_ab * c[inside][:, 2]) / (e_ab + e_bc + e_ca)
        valid = (z.trunc() >= 0) & (z.trunc() < grid)
        point = torch.empty((int(valid.sum()), 3), dtype=torch.float64, device=p.device)
        point[:, ax0], point[:, ax1], point[:, ax2] = p[valid, 0], p[valid, 1], z[valid]
        tri_ids.append(tri[valid])
        axes.append(torch.full_like(tri[valid], ax2))
        points.append(point)
    return torch.cat(tri_ids), torch.cat(axes), torch.cat(points)


def _crossing_voxels(axes, points):
    """The four voxels (C, 4, 3) around each crossing, the first one owning the crossed edge (its intersected flag)."""
    voxels = points.trunc().long().unsqueeze(1).repeat(1, 4, 1)
    for corner, (dx, dy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        ax0, ax1 = (axes + 1) % 3, (axes + 2) % 3
        row = torch.arange(len(axes), device=axes.device)
        voxels[row, corner, ax0] += dx - 1
        voxels[row, corner, ax1] += dy - 1
    return voxels


def _keys(voxels, grid):
    return (voxels[..., 0] * grid + voxels[..., 1]) * grid + voxels[..., 2]


def _face_overlaps(triangles, grid):
    """Voxels overlapped by each triangle (upstream face_qef plane / projection tests): (triangle index, voxel (3,))."""
    v0, v1, v2 = triangles.unbind(1)
    e0, e1, e2 = v1 - v0, v2 - v1, v0 - v2
    n = F.normalize(torch.linalg.cross(e0, e1), dim=-1)
    lo = triangles.amin(1).trunc().long().clamp_min(0)
    hi = ((triangles.amax(1) + 1).trunc().long().clamp(max=grid)) - 1
    rows, voxels = _enumerate(lo, hi)
    p = voxels.float()
    nr, a, b, c, e = n[rows], v0[rows], v1[rows], v2[rows], (e0[rows], e1[rows], e2[rows])
    corner = (nr > 0).float()
    d1 = (nr * (corner - a)).sum(-1)
    d2 = (nr * (1 - corner - a)).sum(-1)
    dot = (nr * p).sum(-1)
    keep = (dot + d1) * (dot + d2) <= 0
    starts = (a, b, c)
    for u, v, w in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):  # XY (normal z), YZ (normal x), ZX (normal y) projections
        mul = torch.where(nr[:, w] < 0, -1.0, 1.0)
        for edge, start in zip(e, starts):
            nu, nv = -mul * edge[:, v], mul * edge[:, u]
            d = -(nu * start[:, u] + nv * start[:, v]) + nu.clamp_min(0) + nv.clamp_min(0)
            keep &= nu * p[:, u] + nv * p[:, v] + d >= 0
    return rows[keep], voxels[keep]


def _boundary_segments(vertices, faces):
    """Edges used by one face only, (S, 2, 3)."""
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).sort(dim=1).values
    unique, counts = torch.unique(edges, dim=0, return_counts=True)
    return vertices[unique[counts == 1]]


def _segment_voxels(segments):
    """Voxels traversed by each segment (upstream 3D DDA, ties stepping z, then y, then x): (segment index, voxel (3,))."""
    v0, v1 = segments[:, 0].double(), segments[:, 1].double()
    length = (v1 - v0).norm(dim=-1)
    keep = (length >= 1e-6).nonzero().squeeze(1)  # upstream skips degenerate edges (world units: 1e-6 is far below a voxel)
    v0, length = v0[keep], length[keep]
    direction = (v1[keep] - v0) / length[:, None]
    start = v0.floor().long()
    step = torch.where(direction > 0, 1, -1)
    border = (start + (step > 0).long()).double()
    t_max = torch.where(direction == 0, torch.inf, (border - v0) / direction)
    t_delta = torch.where(direction == 0, torch.inf, 1 / direction.abs())
    counts = torch.where(t_max <= length[:, None], ((length[:, None] - t_max) / t_delta).floor() + 1, 0).long()
    rows, events = [], []
    for axis in range(3):
        seg, k = _enumerate(torch.zeros_like(counts[:, axis:axis + 1]), counts[:, axis:axis + 1] - 1)
        rows.append(seg)
        events.append(torch.stack([t_max[seg, axis] + k[:, 0] * t_delta[seg, axis], torch.full_like(seg, axis).double()], 1))
    seg, events = torch.cat(rows), torch.cat(events)
    order = torch.argsort(-events[:, 1], stable=True)  # ordered by segment, then crossing distance, ties stepping the higher axis first
    order = order[torch.argsort(events[order, 0], stable=True)]
    order = order[torch.argsort(seg[order], stable=True)]
    seg, axis = seg[order], events[order, 1].long()
    moves = torch.zeros((len(seg), 3), dtype=torch.long, device=seg.device)
    moves[torch.arange(len(seg), device=seg.device), axis] = step[seg, axis]
    totals = moves.cumsum(0)
    first = torch.searchsorted(seg, seg)  # first crossing of each segment
    visited = torch.cat([start, start[seg] + totals - totals[first] + moves[first]])
    return keep[torch.cat([torch.arange(len(keep), device=seg.device), seg])], visited


def _accumulate(target, index, values):
    """target[i] += the sum of the values rows of index i, in a fixed order (index_add_'s atomics add in an order that varies between runs,
    which moved dual vertices by up to 0.05 voxel): rows sorted by index, then summed segment by segment."""
    order = torch.argsort(index, stable=True)
    target += torch.segment_reduce(values[order].to(target.dtype), "sum", lengths=torch.bincount(index, minlength=target.shape[0]), axis=0)


def _solve(quadric, means, counts, voxels, regularization_weight):
    """Dual vertex of each voxel (voxel units): the quadric minimum (with the pull to the crossings' mean), constrained to the voxel."""
    xx, xy, xz, xw, yy, yz, yw, zz, zw, ww = quadric.double().unbind(1)
    Q = torch.stack([torch.stack(r, -1) for r in ((xx, xy, xz, xw), (xy, yy, yz, yw), (xz, yz, zz, zw), (xw, yw, zw, ww))], -2)
    if regularization_weight > 0:
        p = means.double() / counts.double()[:, None]
        w = (regularization_weight * counts.double())[:, None, None]
        reg = torch.zeros_like(Q)
        reg[:, [0, 1, 2], [0, 1, 2]] = 1
        reg[:, :3, 3], reg[:, 3, :3], reg[:, 3, 3] = -p, -p, (p * p).sum(-1)
        Q = Q + w * reg
    low = voxels.double()
    high = low + 1
    v = torch.linalg.solve(Q[:, :3, :3], -Q[:, :3, 3:])[..., 0]
    outside = ~((v >= low) & (v <= high)).all(-1)
    if bool(outside.any()):
        rows = outside.nonzero().squeeze(1)
        Qo, lo, hi = Q[rows], low[rows], high[rows]
        candidates, feasible = [], []

        def add(point, ok):
            candidates.append(point)
            feasible.append(ok)

        for fixed in range(3):
            a1, a2 = (fixed + 1) % 3, (fixed + 2) % 3
            A = Qo[:, [a1, a2]][:, :, [a1, a2]]
            B = Qo[:, [a1, a2]][:, :, [fixed, 3]]
            for bound in (lo, hi):
                x = torch.linalg.solve(A, -(B[..., 0] * bound[:, fixed, None] + B[..., 1]).unsqueeze(-1))[..., 0]
                point = torch.empty_like(lo)
                point[:, fixed], point[:, a1], point[:, a2] = bound[:, fixed], x[:, 0], x[:, 1]
                add(point, (x[:, 0] >= lo[:, a1]) & (x[:, 0] <= hi[:, a1]) & (x[:, 1] >= lo[:, a2]) & (x[:, 1] <= hi[:, a2]))
        for free in range(3):
            a1, a2 = (free + 1) % 3, (free + 2) % 3
            a = Qo[:, free, free]
            for q1, q2 in ((lo, lo), (lo, hi), (hi, lo), (hi, hi)):
                x = -(Qo[:, free, a1] * q1[:, a1] + Qo[:, free, a2] * q2[:, a2] + Qo[:, free, 3]) / a
                point = torch.empty_like(lo)
                point[:, free], point[:, a1], point[:, a2] = x, q1[:, a1], q2[:, a2]
                add(point, (x >= lo[:, free]) & (x <= hi[:, free]))
        for cx in (hi, lo):  # upstream: constraint flag 0 selects the max corner
            for cy in (hi, lo):
                for cz in (hi, lo):
                    add(torch.stack([cx[:, 0], cy[:, 1], cz[:, 2]], -1), torch.ones(len(rows), dtype=torch.bool, device=rows.device))
        points = torch.stack(candidates, 1)
        homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], -1)
        err = torch.einsum("rci,rij,rcj->rc", homogeneous, Qo, homogeneous)
        err = torch.where(torch.stack(feasible, 1), err.nan_to_num(nan=torch.inf), torch.inf)
        v[rows] = points[torch.arange(len(rows), device=rows.device), err.argmin(1)]
    return v.float()


@torch.no_grad()
def mesh_to_flexible_dual_grid(vertices, faces, grid_size, face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2):
    """Flexible dual grid of a mesh whose vertices are in grid units (0 .. grid_size, the bounding box min at 0): active voxel indices
    (N, 3) int64 sorted by x, y, z, dual vertex offsets in the voxels (N, 3) in [0, 1], and intersected edge flags (N, 3) bool (whether
    the voxel's edge along each axis at its max corner crosses the surface). Defaults of TRELLIS.2 / SUPRA-Tex encoding."""
    vertices, faces = vertices.float(), faces.long()
    device = vertices.device
    crossings = [_crossings(vertices[faces[start:start + _TRIANGLES]], grid_size) for start in range(0, len(faces), _TRIANGLES)]
    tri = torch.cat([c[0] + start for c, start in zip(crossings, range(0, len(faces), _TRIANGLES))])
    axes, points = torch.cat([c[1] for c in crossings]), torch.cat([c[2] for c in crossings])
    del crossings
    corner_voxels = _crossing_voxels(axes, points)
    keys, inverse = torch.unique(_keys(corner_voxels, grid_size).reshape(-1), return_inverse=True)
    del corner_voxels
    count = len(keys)
    voxels = torch.stack([keys // (grid_size * grid_size), keys // grid_size % grid_size, keys % grid_size], 1)
    planes = torch.zeros((len(faces), 10), dtype=torch.float32, device=device)
    for start in range(0, len(faces), _TRIANGLES):
        planes[start:start + _TRIANGLES] = _plane_quadrics(vertices[faces[start:start + _TRIANGLES]])
    quadric = torch.zeros((count, 10), dtype=torch.float32, device=device)
    means = torch.zeros((count, 3), dtype=torch.float32, device=device)
    counts = torch.zeros(count, dtype=torch.float32, device=device)
    inverse = inverse.view(-1, 4)
    for corner in range(4):
        _accumulate(quadric, inverse[:, corner], planes[tri])
        _accumulate(means, inverse[:, corner], points.float())
        counts.index_add_(0, inverse[:, corner], torch.ones(len(tri), dtype=torch.float32, device=device))
    intersected = torch.zeros((count, 3), dtype=torch.bool, device=device)
    intersected[inverse[:, 0], axes] = True
    del tri, axes, points, inverse

    def existing(candidate_voxels):
        found = torch.searchsorted(keys, _keys(candidate_voxels, grid_size)).clamp_(max=count - 1)
        return found, keys[found] == _keys(candidate_voxels, grid_size)

    if face_weight > 0:
        for start in range(0, len(faces), _TRIANGLES):
            rows, overlapped = _face_overlaps(vertices[faces[start:start + _TRIANGLES]], grid_size)
            found, hit = existing(overlapped)
            _accumulate(quadric, found[hit], planes[start + rows[hit]] * face_weight)
    if boundary_weight > 0:
        segments = _boundary_segments(vertices, faces)
        if len(segments):
            seg, visited = _segment_voxels(segments)
            inside = ((visited >= 0) & (visited < grid_size)).all(-1)
            found, hit = existing(visited.clamp(0, grid_size - 1))
            hit &= inside
            v0, v1 = segments[:, 0].double(), segments[:, 1].double()
            direction = F.normalize(v1 - v0, dim=-1)
            A = torch.eye(3, dtype=torch.float64, device=device) - direction[:, :, None] * direction[:, None, :]
            b = -(A @ v0[:, :, None])[..., 0]
            c = (v0 * (A @ v0[:, :, None])[..., 0]).sum(-1)
            edge = torch.stack([A[:, 0, 0], A[:, 0, 1], A[:, 0, 2], b[:, 0], A[:, 1, 1], A[:, 1, 2], b[:, 1], A[:, 2, 2], b[:, 2], c], 1).float()
            _accumulate(quadric, found[hit], edge[seg[hit]] * boundary_weight)
    dual = torch.empty((count, 3), dtype=torch.float32, device=device)
    for start in range(0, count, _SOLVE_ROWS):
        part = slice(start, start + _SOLVE_ROWS)
        dual[part] = _solve(quadric[part], means[part], counts[part], voxels[part], regularization_weight)
    return voxels, (dual - voxels).clamp_(0, 1), intersected
