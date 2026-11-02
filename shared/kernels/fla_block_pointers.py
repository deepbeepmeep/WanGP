# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Lower FLA's block accesses to masked pointers before Triton 3.9 compilation.

Only the selected, cloned inference kernels are changed; neither package is
edited on disk and Triton's API is untouched. This follows FLA upstream's
block-pointer migration (fla-org/flash-linear-attention#1062), retaining the
installed kernel's arithmetic, tiling and launch configuration.
"""
import ast


def _tl_call(node, name):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl" and node.func.attr == name


class _MaskedPointers(ast.NodeTransformer):
    def __init__(self, tree):
        self.pointers = {node.targets[0].id for node in ast.walk(tree) if isinstance(node, ast.Assign) and _tl_call(node.value, "make_block_ptr")}

    def visit_Assign(self, node):
        if not _tl_call(node.value, "make_block_ptr"):
            return self.generic_visit(node)
        name = node.targets[0].id
        args = dict(zip(("base", "shape", "strides", "offsets", "block_shape", "order"), node.value.args))
        args.update({kw.arg: kw.value for kw in node.value.keywords})
        shape, strides, offsets, blocks = (args[key].elts for key in ("shape", "strides", "offsets", "block_shape"))
        rank = len(blocks)
        statements, address = [], ast.unparse(args["base"])
        for axis in range(rank):
            index = f"_wgp_{name}_i{axis}"
            mask = f"_wgp_{name}_m{axis}"
            # Block-pointer addressing uses signed 64-bit strides. Keep that
            # contract for large/strided tensors; the compiler folds constants.
            expand = ", ".join(":" if i == axis else "None" for i in range(rank))
            statements.append(f"{index} = {ast.unparse(offsets[axis])} + tl.arange(0, {ast.unparse(blocks[axis])})")
            statements.append(f"{mask} = (({index} >= 0) & ({index} < {ast.unparse(shape[axis])}))[{expand}]")
            address = f"({address}) + tl.cast({index}[{expand}], tl.int64) * tl.cast({ast.unparse(strides[axis])}, tl.int64)"
        statements.append(f"{name} = {address}")
        return ast.parse("\n".join(statements)).body

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if not (_tl_call(node, "load") or _tl_call(node, "store")):
            return node
        pointer = node.args[0] if node.args else next(kw.value for kw in node.keywords if kw.arg == "pointer")
        if not isinstance(pointer, ast.Name) or pointer.id not in self.pointers:
            return node
        keywords = {kw.arg: kw.value for kw in node.keywords}
        boundary = keywords.pop("boundary_check", ast.Tuple(elts=[]))
        padding = keywords.pop("padding_option", ast.Constant(value=""))
        if boundary.elts:
            masks = [f"_wgp_{pointer.id}_m{axis.value}" for axis in boundary.elts]
            keywords["mask"] = ast.parse(" & ".join(masks), mode="eval").body
            if _tl_call(node, "load"):
                # FLA's unpadded boundary loads also rely on zero-filled lanes.
                keywords["other"] = ast.parse('float("nan")' if padding.value == "nan" else "0", mode="eval").body
        node.keywords = [ast.keyword(arg=name, value=value) for name, value in keywords.items()]
        return node


def lower_block_pointers(kernel):
    """Rewrite a fresh JITFunction, before callers or cache keys can reference it."""
    tree = ast.parse(kernel.src)
    if not any(_tl_call(node, "make_block_ptr") for node in ast.walk(tree)):
        return kernel
    tree = _MaskedPointers(tree).visit(tree)
    ast.fix_missing_locations(tree)
    if any(_tl_call(node, "make_block_ptr") or _tl_call(node, "advance") for node in ast.walk(tree)):
        raise ValueError("Unsupported block-pointer expression in the FLA inference kernel")
    kernel._unsafe_update_src(ast.unparse(tree) + "\n")
    return kernel
