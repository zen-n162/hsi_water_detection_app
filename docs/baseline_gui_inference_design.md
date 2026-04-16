# Baseline GUI Inference Design

## Scope

This document describes the minimal design needed to add the baseline v3 reference model to the GUI path without replacing the production HyperSIGMA default.

## Backend Files To Extend

- `backend/app/api/routes_inference.py`
- `backend/app/services/inference_service.py`
- `backend/app/services/deploy_config_service.py`
- likely a new service such as `backend/app/services/baseline_inference_service.py`

## Frontend Files To Extend

- `frontend/src/App.tsx`
- `frontend/src/lib/api.ts`

## Deploy Config Strategy

Keep the current HyperSIGMA deploy config as the production default and add a separate baseline deploy profile for the reference path.

- HyperSIGMA deploy config should stay unchanged.
- Baseline should use its own deploy config rather than overloading the current one with incompatible fields.

## API Contract Difference

HyperSIGMA path today returns attention artifacts and uses calibration inputs.

Baseline path would differ in these ways:

- no spectral attention artifact
- no spatial attention artifact unless a baseline-specific visualization is added
- likely no `temperature_json`
- same provenance shape should still be preserved

## Provenance Unification

Both paths should return the same top-level provenance keys:

- `resolved_model_checkpoint`
- `resolved_threshold`
- `resolved_manifest`
- `resolved_dataset`
- `resolved_run_name`
- `requested_device`
- `executed_device`

Fields that do not apply should be `null` rather than omitted.

## Breakage Risks

- Mixing HyperSIGMA-specific and baseline-specific logic in one route can make the current production path fragile.
- Reusing one deploy config for both paths can blur which artifacts are truly production defaults.
- Frontend UI may become confusing if model-family differences are not labeled clearly.

## Minimal Implementation Order

1. Add a separate baseline deploy config and backend inference service.
2. Extend the API so the frontend can request `model_family=baseline` without changing the HyperSIGMA default path.
3. Return baseline provenance in the same shape as HyperSIGMA provenance.
4. Add a small frontend selector for production HyperSIGMA vs reference baseline.
5. Add a baseline-specific smoke test before exposing it as a supported reference mode.
