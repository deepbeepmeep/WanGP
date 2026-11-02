# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from nv-tlabs/FlexiCubes. Every change is marked
# below with a "SPARC3D modification" comment. In summary:
#
#   * sparse cube indexing -- ``_get_case_id_sparse`` resolves the ambiguous DMC
#     configurations through an explicit ``cube_index_map`` instead of scanning a
#     dense ``res x res x res`` array, so the extractor never materializes the
#     full grid;
#   * lookup tables and index tensors are int32 rather than int64;
#   * ``__call__`` takes a ``dtype`` so the extractor can run in half precision;
#   * the quadrilateral winding fix is out-of-place (``torch.cat``) instead of
#     masked in-place assignment;
#   * removed upstream code that this inference-only package never reaches: the
#     ``construct_voxel_grid`` dense grid builder, ``_tetrahedralize`` and the
#     ``output_tetmesh`` output mode, the ``grad_func`` gradient path, the dense
#     ``_get_case_id`` superseded by ``_get_case_id_sparse``, the unused
#     ``_solve_vd_QEF`` dual-vertex solver, the ``L_dev`` developability
#     regularizer, and the differentiable ``training`` quad-splitting branch;
#   * removed the learned per-cube weights ``alpha``/``beta``/``gamma_f`` and the
#     ``weight_scale`` that normalizes them. They are trainable features of the
#     upstream extractor and no released checkpoint carries them, so the
#     inference path always passed ``None`` and ``_normalize_weights`` always
#     returned tensors of ones. Folding those ones through leaves plain linear
#     interpolation, an unweighted mean over each dual vertex, and the fixed
#     ``quad_split_2`` triangulation of every quadrilateral. The float32
#     accumulation the all-ones weights used to force is kept explicitly, so
#     half-precision inputs still reduce in float32;
#   * ``cube_index_map`` is required rather than defaulting to ``None``, and the
#     unread ``cube_corners`` table is dropped.
#
# Vendored for WanGP from hi3d-official/Hi3Dv1.0 (sparc3d/ops/flexicubes). Modifications by DeepBeepMeep, marked "WanGP modification":
# the surface edges are deduplicated as int64 keys (same order as the row-wise unique, a fraction of its memory) and every table
# is created on the extractor's device.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""FlexiCubes dual marching cubes extractor with sparse cube indexing."""

import torch

# SPARC3D modification: relative import, since the tables live in this package.
from .flexicubes_tables import *

__all__ = [
    'FlexiCubes'
]


class FlexiCubes:
    """Dual marching cubes extractor over a sparse set of occupied cubes."""

    # SPARC3D modification: every table below is int32 rather than upstream's int64.
    def __init__(self, device="cuda"):
        """
        Args:
            device: Device the lookup tables are allocated on. Inputs to
                :meth:`__call__` must live on the same device.
        """
        self.device = device
        self.dmc_table = torch.tensor(dmc_table, dtype=torch.int32, device=device, requires_grad=False)
        self.num_vd_table = torch.tensor(num_vd_table,
                                         dtype=torch.int32, device=device, requires_grad=False)
        self.check_table = torch.tensor(
            check_table,
            dtype=torch.int32, device=device, requires_grad=False)

        self.quad_split_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.int32, device=device, requires_grad=False)

        # WanGP modification: on the extractor's device
        self.cube_corners_idx = torch.pow(2, torch.arange(8, dtype=torch.int32, device=device))
        self.cube_edges = torch.tensor([0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6,
                                       2, 0, 3, 1, 7, 5, 6, 4], dtype=torch.int32, device=device, requires_grad=False)
        # WanGP modification: per cube edge, the rank of its second vertex among the possible ones of its first vertex (-y, +z, +x)
        self.edge_axis_code = torch.tensor([2, 1, 2, 1, 2, 1, 2, 1, 0, 0, 0, 0], dtype=torch.int64, device=device)

    # SPARC3D modification: renamed arguments, added dtype and cube_index_map, and
    # dropped upstream's output_tetmesh and grad_func. Extracts a triangle mesh from the
    # sparse scalar field; returns (vertices, faces).
    def __call__(self, voxelgrid_vertices, scalar_field, cube_idx, resolution,
                 cube_index_map, dtype=None): # dtype so that it can work with half precision.
        """Extract a triangle mesh from a sparse signed scalar field.

        Args:
            voxelgrid_vertices: Cube-corner positions ``[V, 3]``.
            scalar_field: Signed value at each corner ``[V]``; the surface is its
                zero level set.
            cube_idx: Corner indices of each occupied cube ``[C, 8]``.
            resolution: Grid resolution the cube indices are expressed in.
            cube_index_map: Integer grid coordinate of each cube ``[C, 3]``, used to
                resolve ambiguous configurations against neighbours. Required: the
                sparse case-id path indexes it directly, so omitting it fails only
                once a mesh happens to contain an ambiguous cube.
            dtype: Float dtype to unify inputs to, e.g. ``torch.float16``. Inferred
                from ``voxelgrid_vertices`` when ``None``, defaulting to float32.

        Returns:
            ``(vertices, faces)``, both empty when no cube straddles the surface.
        """
        # unify floating tensors to a chosen dtype, so it can work with half, etc:
        if dtype is None:
            if voxelgrid_vertices.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                dtype = voxelgrid_vertices.dtype
            else:
                dtype = torch.float32

        # cast all to dtype, so that it works with half-precision:
        voxelgrid_vertices = voxelgrid_vertices.to(dtype)  # Cast the geometry to the chosen dtype
        scalar_field = scalar_field.to(dtype)

        surf_cubes, occ_fx8 = self._identify_surf_cubes(scalar_field, cube_idx)
        if surf_cubes.sum() == 0:
            return (
                torch.zeros((0, 3), device=self.device, dtype=dtype),
                torch.zeros((0, 3), dtype=torch.int32, device=self.device),
            )

        case_ids = self._get_case_id_sparse(occ_fx8, surf_cubes, resolution, cube_index_map)
        
        surf_edges, idx_map, edge_counts, surf_edges_mask = self._identify_surf_edges(
            scalar_field, cube_idx, surf_cubes
        )

        vd, vd_idx_map = self._compute_vd(
            voxelgrid_vertices, surf_edges, scalar_field, case_ids, idx_map)

        return self._triangulate(
            scalar_field, surf_edges, vd, edge_counts, idx_map,
            vd_idx_map, surf_edges_mask)

    # SPARC3D modification: sparse replacement for _get_case_id. Upstream builds a
    # dense res^3 x 5 array to look up the configuration of adjacent cubes; this
    # version sorts the occupied cube keys and binary-searches them instead, so
    # cost scales with the number of surface cubes rather than with res^3.
    @torch.no_grad()
    def _get_case_id_sparse(self, occ_fx8, surf_cubes, res, cube_index_map):
        """
        Obtains the ID of topology cases based on cell corner occupancy. This function resolves the 
        ambiguity in the Dual Marching Cubes (DMC) configurations as described in Section 1.3 of the 
        supplementary material.
        """
        occ_int = occ_fx8[surf_cubes].to(torch.int32)  # Convert boolean to int32
        corners_idx = self.cube_corners_idx.to(self.device).unsqueeze(0)
        case_ids = (occ_int * corners_idx).sum(-1, dtype=torch.int32) # int32 else sum defaults to int64 by default
        problem_config = self.check_table.to(self.device)[case_ids]
        to_check = problem_config[..., 0] == 1
        
        problem_config = problem_config[to_check]
        problem_config_index = cube_index_map[surf_cubes][to_check]

        # SPARC3D modification: an ambiguous cube is resolved against its neighbour, which
        # upstream looks up in a dense res^3 array of per-cube configurations. Here the
        # neighbour is found by binary-searching the sorted keys of the occupied cubes
        # below, so no dense volume is ever allocated.
        vol_idx = cube_index_map  # N, 3
        vol_idx_problem = vol_idx[surf_cubes][to_check]
        
        vol_idx_problem_adj = vol_idx_problem + problem_config[..., 1:4]

        within_range = (
            vol_idx_problem_adj[..., 0] >= 0) & (
            vol_idx_problem_adj[..., 0] < res) & (
            vol_idx_problem_adj[..., 1] >= 0) & (
            vol_idx_problem_adj[..., 1] < res) & (
            vol_idx_problem_adj[..., 2] >= 0) & (
            vol_idx_problem_adj[..., 2] < res)

        vol_idx_problem = vol_idx_problem[within_range]
        vol_idx_problem_adj = vol_idx_problem_adj[within_range]
        problem_config = problem_config[within_range]
        problem_config_index = problem_config_index[within_range]

        if problem_config.shape[0] == 0:
            return case_ids

        inverse_keys = (
            (vol_idx_problem_adj[..., 0].long() * res
             + vol_idx_problem_adj[..., 1].long()) * res
            + vol_idx_problem_adj[..., 2].long()
        )
        problem_keys = (
            (problem_config_index[..., 0].long() * res
             + problem_config_index[..., 1].long()) * res
            + problem_config_index[..., 2].long()
        )
        sorted_keys, sorted_order = torch.sort(problem_keys)
        positions = torch.searchsorted(sorted_keys, inverse_keys)
        bounded_positions = positions.clamp(max=sorted_keys.shape[0] - 1)
        found = (positions < sorted_keys.shape[0]) & (
            sorted_keys[bounded_positions] == inverse_keys
        )
        inverse_indices = torch.full_like(positions, -1, dtype=torch.long)
        inverse_indices[found] = sorted_order[bounded_positions[found]]
        problem_config = torch.cat(
            (
                problem_config,
                torch.zeros(
                    (1, 5),
                    dtype=problem_config.dtype,
                    device=problem_config.device,
                ),
            )
        )
        problem_config_adj = problem_config[inverse_indices]

        # If two cubes with cases C16 and C19 share an ambiguous face, both cases are inverted.
        to_invert = (problem_config_adj[..., 0] == 1)
        idx = torch.arange(case_ids.shape[0], dtype=torch.int32, device=self.device)[to_check][within_range][to_invert]
        case_ids.index_put_((idx,), problem_config[:-1][to_invert][..., -1])

        return case_ids
    

    @torch.no_grad()
    def _identify_surf_edges(self, scalar_field, cube_idx, surf_cubes):
        """
        Identifies grid edges that intersect with the underlying surface by checking for opposite signs. As each edge 
        can be shared by multiple cubes, this function also assigns a unique index to each surface-intersecting edge 
        and marks the cube edges with this index.
        """
        occ_n = scalar_field < 0
        # WanGP modification: the unique edges and their per-cube inverse from a dense table indexed by (first vertex, axis) instead
        # of a sort of every (first, second) vertex pair; the axis codes follow the order of the second vertices (vertices are sorted
        # by x, then y, then z: the -y neighbour comes first, then +z, then +x), so the unique edges keep upstream's order.
        cubes = cube_idx[surf_cubes]
        first, second = cubes[:, self.cube_edges[0::2]], cubes[:, self.cube_edges[1::2]]
        del cubes
        keys = (first.long() * 3 + self.edge_axis_code).view(-1)
        del first
        size = scalar_field.shape[0] * 3
        counts_dense = torch.bincount(keys, minlength=size).to(torch.int32)
        present = counts_dense > 0
        ids_dense = torch.cumsum(present, 0, dtype=torch.int32) - 1
        second_dense = torch.empty(size, dtype=torch.int32, device=keys.device)
        second_dense[keys] = second.reshape(-1).to(torch.int32)
        del second
        unique_keys = present.nonzero().squeeze(1)
        del present
        unique_edges = torch.stack([(unique_keys // 3).to(torch.int32), second_dense[unique_keys]], dim=1)
        del unique_keys, second_dense
        _idx_map, counts = ids_dense[keys], counts_dense[keys]
        del keys, ids_dense, counts_dense

        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1

        surf_edges_mask = mask_edges[_idx_map]

        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.int32, device=cube_idx.device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.int32, device=cube_idx.device) # int32 else arrange defaults to int64
        # Shaped as [number of cubes x 12 edges per cube]. This is later used to map a cube edge to the unique index
        # for a surface-intersecting edge. Non-surface-intersecting edges are marked with -1.
        idx_map = mapping[_idx_map]
        surf_edges = unique_edges[mask_edges]
        return surf_edges, idx_map, counts, surf_edges_mask

    @torch.no_grad()
    def _identify_surf_cubes(self, scalar_field, cube_idx):
        """
        Identifies grid cubes that intersect with the underlying surface by checking if the signs at 
        all corners are not identical.
        """
        occ_n = scalar_field < 0
        occ_fx8 = occ_n[cube_idx.reshape(-1)].reshape(-1, 8)
        _occ_sum = torch.sum(occ_fx8, -1)
        surf_cubes = (_occ_sum > 0) & (_occ_sum < 8)
        return surf_cubes, occ_fx8

    def _linear_interp(self, edges_weight, edges_x):
        """
        Computes the location of zero-crossings on 'edges_x' using linear interpolation with 'edges_weight'.
        """
        edge_dim = edges_weight.dim() - 2
        assert edges_weight.shape[edge_dim] == 2
        edges_weight = torch.cat([torch.index_select(input=edges_weight, index=torch.tensor(1, device=self.device), dim=edge_dim), -
                                 torch.index_select(input=edges_weight, index=torch.tensor(0, device=self.device), dim=edge_dim)]
                                 , edge_dim)
        denominator = edges_weight.sum(edge_dim)
        ue = (edges_x * edges_weight).sum(edge_dim) / denominator
        return ue

    def _compute_vd(self, voxelgrid_vertices, surf_edges, scalar_field,
                    case_ids, idx_map):
        """
        Computes the location of dual vertices as described in Section 4.2
        """
        surf_edges_x = torch.index_select(input=voxelgrid_vertices, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, 3)
        surf_edges_s = torch.index_select(input=scalar_field, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, 1)

        idx_map = idx_map.reshape(-1, 12)
        num_vd = torch.index_select(input=self.num_vd_table, index=case_ids, dim=0)
        edge_group, edge_group_to_vd, edge_group_to_cube = [], [], []

        total_num_vd = 0
        # WanGP modification: int32 index arrays, the edge crossings released once averaged, and the cube-edge to dual-vertex map
        # scattered in place (upstream's v_idx[edge_group_to_vd] is edge_group_to_vd itself).
        for num in torch.unique(num_vd).tolist():
            cur_cubes = (num_vd == num)  # consider cubes with the same numbers of vd emitted (for batching)
            curr_num_vd = int(cur_cubes.sum()) * num
            curr_edge_group = self.dmc_table[case_ids[cur_cubes], :num].reshape(-1, num * 7)
            curr_edge_group_to_vd = (torch.arange(curr_num_vd, dtype=torch.int32, device=self.device) + total_num_vd).unsqueeze(-1).expand(-1, 7).reshape_as(curr_edge_group)
            total_num_vd += curr_num_vd
            curr_edge_group_to_cube = cur_cubes.nonzero().squeeze(1).to(torch.int32).unsqueeze(-1).expand(-1, num * 7)

            curr_mask = (curr_edge_group != -1)
            edge_group.append(curr_edge_group[curr_mask])
            edge_group_to_vd.append(curr_edge_group_to_vd[curr_mask])
            edge_group_to_cube.append(curr_edge_group_to_cube[curr_mask])
            del cur_cubes, curr_edge_group, curr_edge_group_to_vd, curr_edge_group_to_cube, curr_mask

        edge_group = torch.cat(edge_group)
        edge_group_to_vd = torch.cat(edge_group_to_vd)
        index = torch.cat(edge_group_to_cube).long() * 12 + edge_group
        del edge_group, edge_group_to_cube

        # SPARC3D modification: dual vertices accumulate in float32 even when the inputs
        # are half. Averaging up to twelve edge crossings per vertex in half precision
        # loses enough mantissa to move the surface visibly.
        idx_group = torch.gather(input=idx_map.reshape(-1), dim=0, index=index)
        x_group = torch.index_select(input=surf_edges_x, index=idx_group, dim=0).reshape(-1, 2, 3)
        s_group = torch.index_select(input=surf_edges_s, index=idx_group, dim=0).reshape(-1, 2, 1)
        del idx_group, surf_edges_x, surf_edges_s
        ue_group = self._linear_interp(s_group.float(), x_group)
        del x_group, s_group

        edges_per_vd = torch.zeros((total_num_vd, 1), device=self.device, dtype=torch.float).index_add_(
            0, index=edge_group_to_vd, source=torch.ones_like(ue_group[:, :1]))
        vd = torch.zeros((total_num_vd, 3), device=self.device, dtype=torch.float).index_add_(0, index=edge_group_to_vd, source=ue_group).div_(edges_per_vd)
        del ue_group, edges_per_vd

        vd_idx_map = torch.zeros(case_ids.shape[0] * 12, dtype=torch.int32, device=self.device).scatter_(0, index, edge_group_to_vd)

        return vd, vd_idx_map

    # SPARC3D modification: every quadrilateral splits along the same fixed diagonal.
    # Choosing per-quad between the two diagonals needs the learned gamma weights, which
    # the released inference weights do not carry.
    def _triangulate(self, scalar_field, surf_edges, vd, edge_counts, idx_map, vd_idx_map, surf_edges_mask):
        """
        Connects four neighboring dual vertices to form a quadrilateral. Each quadrilateral is then split
        into two triangles along a fixed diagonal, as described in Section 4.3.
        """
        with torch.no_grad():
            group_mask = (edge_counts == 4) & surf_edges_mask  # surface edges shared by 4 cubes.
            group = idx_map.reshape(-1)[group_mask]
            vd_idx = vd_idx_map[group_mask]
            edge_indices, indices = torch.sort(group, stable=True)
            indices = indices.to(torch.int32)
            quad_vd_idx = vd_idx[indices].reshape(-1, 4).to(torch.int32)

            # Ensure all face directions point towards the positive SDF to maintain consistent winding.
            # SPARC3D modification: rebuilt out-of-place rather than by masked
            # in-place assignment, which reorders the quadrilaterals.
            s_edges = scalar_field[surf_edges[edge_indices.reshape(-1, 4)[:, 0]].reshape(-1)].reshape(-1, 2)
            flip_mask = s_edges[:, 0] > 0
            quad_vd_idx = torch.cat((quad_vd_idx[flip_mask][:, [0, 1, 3, 2]],
                                     quad_vd_idx[~flip_mask][:, [2, 3, 1, 0]]))

        faces = quad_vd_idx[:, self.quad_split_2].reshape(-1, 3)
        return vd, faces