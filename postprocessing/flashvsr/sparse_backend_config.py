from __future__ import annotations

SPARSE_BACKEND_AUTO = "auto"
SPARSE_BACKEND_TRITON_SPARSE = "triton_sparse"
SPARSE_BACKEND_SPARGE = "sparge"

SPARSE_BACKEND_LABELS = {
    SPARSE_BACKEND_AUTO: "Auto",
    SPARSE_BACKEND_TRITON_SPARSE: "Triton Sparse Attention",
    SPARSE_BACKEND_SPARGE: "Installed Sparge kernels (SparseGP or SpargeAttention)",
}
SPARSE_BACKEND_CHOICES = [
    (SPARSE_BACKEND_LABELS[key], key)
    for key in (SPARSE_BACKEND_AUTO, SPARSE_BACKEND_TRITON_SPARSE, SPARSE_BACKEND_SPARGE)
]
SUPPORTED_SPARSE_BACKENDS = frozenset(SPARSE_BACKEND_LABELS)


def normalize_sparse_backend(backend: object) -> str:
    backend = str(backend or SPARSE_BACKEND_AUTO).strip().lower()
    return backend if backend in SUPPORTED_SPARSE_BACKENDS else SPARSE_BACKEND_AUTO
