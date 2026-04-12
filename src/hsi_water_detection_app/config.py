from __future__ import annotations

# Hyperion documentation is usually 1-based band numbering.
# Internal NumPy indexing uses 0-based band indices.

HYPERION_BAD_BANDS_1BASED = (
    list(range(1, 8)) +      # B001-B007
    list(range(58, 77)) +    # B058-B076
    list(range(225, 243))    # B225-B242
)

HYPERION_BAD_BANDS_0BASED = [b - 1 for b in HYPERION_BAD_BANDS_1BASED]

DEFAULT_PATCH_SIZE = 64
DEFAULT_STRIDE = 32
