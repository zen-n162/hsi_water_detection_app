# Next Research Backlog

This backlog starts after the production freeze and must not overwrite the calibrated HyperSIGMA v3 deploy profile until a future candidate passes the same freeze process.

## P1: Parallel Improvements Without Breaking Production

- Baseline GUI path so the reference model can be exercised without changing the production HyperSIGMA route.
- Better calibration beyond temperature scaling, as a sidecar evaluation path first.
- Score collapse diagnostics during fine-tuning so unstable candidates are caught before GUI consideration.
- Richer saved uncertainty and calibration metadata for research-only result inspection.

## P2: Redesign-Level Improvements

- Stronger spatial split with larger block separation or scene-level partitioning.
- Threshold policy redesign that is less brittle than single-point F1 tuning.
- Revisit calibration and threshold jointly under the stronger split.
- Explore class-balanced or confidence-aware fine-tuning losses under the revised evaluation policy.
