# Task 10 Report: 工作流、模型注册与预测实验室前端

## RED / GREEN

- Workflow RED: `npm test -- --run src/modules/agent-workflow/AgentWorkflowView.test.ts` produced 4 expected failures for the absent task-type selector/SSE argument, generic metric display, suggested-vs-final resolution UI, correction/diagnostic rendering, and legacy/classification branches.
- Workflow GREEN: added automatic and explicit task selection, passed it to `streamTraining`, rendered generic/legacy metrics and canonical metadata/corrections, and split regression from classification prediction fields. The initial 4 tests passed.
- Registry RED: `npm test -- --run src/modules/model-registry/ModelRegistryView.test.ts` produced 3 expected failures for regression primary metrics, legacy/malformed metadata markers, accessible refresh, and stale data after a failed reload.
- Registry GREEN: implemented task-aware metric selection and comparison columns, explicit legacy/metadata-error states, invalid comparison disabling, and clear/retry behavior. All 3 tests passed without Element Plus warnings.
- Prediction RED: `npm test -- --run src/modules/prediction-lab/PredictionLabView.test.ts` produced 5 expected failures for missing diagnostics, regression-safe single/batch views, classification compatibility, accessible controls, invalid-model filtering, and stale cleanup.
- Prediction GREEN: implemented the single/batch task branches, diagnostic displays, compatible model filtering, and recovery cleanup. All initial 5 tests passed.
- Self-review RED/GREEN: a running-SSE metadata test failed until event metadata was merged into the visible task state; two deferred-response tests failed until single and batch request-generation guards discarded responses invalidated by model/input changes. A configured `optimization_metric: 'mae'` registry test failed until the primary metric priority consumed it before fallback metrics.

## Files

- Added: `frontend/src/modules/agent-workflow/AgentWorkflowView.test.ts`
- Modified: `frontend/src/modules/agent-workflow/views/AgentWorkflowView.vue`
- Modified: `frontend/src/modules/model-registry/ModelRegistryView.test.ts`
- Modified: `frontend/src/modules/model-registry/views/ModelRegistryView.vue`
- Modified: `frontend/src/modules/prediction-lab/PredictionLabView.test.ts`
- Modified: `frontend/src/modules/prediction-lab/views/PredictionLabView.vue`

## Exact verification

Executed from `frontend` during the final fresh pass:

```text
npm test -- --run
# 21 test files passed, 73 tests passed

npm run typecheck
# vue-tsc --noEmit exited 0

npm run build
# vue-tsc --noEmit and vite build exited 0; 1648 modules transformed
```

The task-specified subset was also run before the self-review fixes:

```text
npm test -- --run src/modules/agent-workflow/AgentWorkflowView.test.ts src/modules/model-registry/ModelRegistryView.test.ts src/modules/prediction-lab/PredictionLabView.test.ts src/modules/interaction-regression.test.ts
# 4 test files passed, 17 tests passed

npm run typecheck
# vue-tsc --noEmit exited 0
```

Focused post-review runs passed with 5 workflow tests, 3 registry tests, and 7 prediction tests. `git diff --check` exited 0.

## Self-review

- Task selection/submission: the workflow selector defaults to `auto`, exposes all four canonical values, and sends the rendered value in the SSE request; tests cover both automatic and explicit regression submissions.
- Resolution/corrections: workflow status and live SSE events show Agent suggestion, canonical final type, source, reason, explanation, target facts, and each structured correction's field/suggested/resolved/reason values.
- Metrics: workflow uses `best_metric`/`best_value` and only falls back to legacy `best_auc`; known metric names receive readable labels. Registry classification priority remains ROC-AUC/val_auc, legacy best_auc, then Accuracy; regression prioritizes the recorded optimization/best metric, then R²/RMSE/MAE, while comparison shows all three regression metrics.
- Registry validity: legacy records have an explicit badge/compatibility label. Malformed schema-v3 records have a distinct error badge, field/message, unavailable task type, and disabled comparison selection.
- Single and batch prediction: regression shows continuous values and per-request/per-row diagnostics while hiding encoded value, probability, positive rate, and average probability. Classification retains encoded value, probability, positive rate, and average probability.
- State lifecycle: model/status reload failures clear stale state and support retries. New workflow tasks, model changes, input edits, CSV replacements, and failed predictions invalidate prior results. Late single/batch responses cannot repopulate a result for a newer selection.
- Accessibility: task/model selectors, refresh buttons, JSON and CSV inputs have accessible labels; live result/diagnostic regions use `aria-live` or status semantics; native buttons and table headings remain available.

## Concerns

- No Task 10 blocker. The production build emitted Rollup notices about misplaced `/* #__PURE__ */` annotations inside the third-party `@vueuse/core` package; Rollup removed those annotations and the build completed successfully.

## Review round 1 — RED / GREEN

- Persisted metric contract RED: the correct project virtualenv run of `tests/test_model_registry_api.py -k "lists_saved_models_with_sources or exposes_only_persisted_metric_identity"` failed 2 tests because registry records omitted canonical metric identity. GREEN: `/api/models` now exposes nullable top-level `optimization_metric` and `best_metric`, populated only from persisted artifact fields (`optimization_metric`/persisted AutoML decision summary and workflow `metric`/`best_metric`); both focused tests passed. Legacy and malformed-v3 records retain null identity and Task 6 validation remains authoritative.
- Frontend contract RED: `npm run typecheck` failed with four missing-property errors after type assertions and realistic API fixtures required those two fields. GREEN: `ModelRegistryRecord` includes both nullable fields and the API contract test passes.
- Registry RED: component tests first received RMSE instead of persisted MAE, then received a fabricated R² when identity was null, and finally omitted the `val_auc` maximize direction in the comparison cell. GREEN: primary metrics consume only persisted identity (legacy compatibility is the sole fallback), all primary/comparison values read the exact capability direction, and four table-cell-scoped tests pass.
- Workflow RED: four rendered tests failed for metric/value-only SSE events, classification fields after a regression workflow, task-type changes that left an active stream/state behind, and callbacks from an aborted task mutating its replacement. GREEN: canonical `metric/value` wins over compatibility best fields; classification response nullability controls its fields; selector changes abort and clear dependent state; captured task/context/stream generations reject all late callbacks. Nine workflow component tests pass.
- Prediction RED: three rendered tests failed because valid legacy models were filtered, older CSV reads overwrote newer input, and a batch response could land after a replacement file was selected. GREEN: only `metadata_error` records are unavailable, legacy prediction remains classification-compatible, CSV selection invalidates immediately, and independent file-read/batch generations reject stale work. Nine prediction tests pass.

## Review round 1 — files

- Modified backend contract/tests: `backend/api/models.py`, `tests/test_model_registry_api.py`.
- Modified shared frontend contract/tests: `frontend/src/api/models.ts`, `frontend/src/api/models.test.ts`.
- Modified workflow: `frontend/src/modules/agent-workflow/views/AgentWorkflowView.vue`, `frontend/src/modules/agent-workflow/AgentWorkflowView.test.ts`.
- Modified registry: `frontend/src/modules/model-registry/views/ModelRegistryView.vue`, `frontend/src/modules/model-registry/ModelRegistryView.test.ts`.
- Modified prediction: `frontend/src/modules/prediction-lab/views/PredictionLabView.vue`, `frontend/src/modules/prediction-lab/PredictionLabView.test.ts` (the full prior classification/interaction suite remains present).

## Review round 1 — exact final verification

```text
/Users/rgj/PycharmProjects/Machine_Learning_Project/.venv/bin/python -m pytest -o addopts='' -q tests/test_model_registry_api.py
# 23 passed in 4.73s

cd frontend && npm test -- --run src/modules/agent-workflow/AgentWorkflowView.test.ts src/modules/model-registry/ModelRegistryView.test.ts src/modules/prediction-lab/PredictionLabView.test.ts src/modules/interaction-regression.test.ts
# 4 files passed, 27 tests passed

cd frontend && npm test -- --run
# 21 files passed, 80 tests passed

cd frontend && npm run typecheck
# vue-tsc --noEmit exited 0

cd frontend && npm run build
# vue-tsc --noEmit and vite build exited 0; 1648 modules transformed
```

## Review round 1 — self-review

- Task selection/submission and replacement: `auto`/user override is submitted exactly; task-type changes and replacement tasks abort prior streams, clear resolved metadata/live metric/events/selected prediction state, and generation-check late events, refreshes, and model selection.
- Workflow rendering: live canonical `metric/value` is preferred; `best_metric/best_value` is compatibility/status fallback and `best_auc` is legacy-only. Suggested/final type, source, reason, explanation, and structured corrections remain visible. Regression responses omit nullable category fields; a selected classification model still renders non-null encoded/probability fields after a regression workflow.
- Registry: primary metric is the exact persisted identity, null identity displays no fabricated primary; legacy `best_auc` remains compatible. Comparison assertions target row cells, and minimize/maximize directions—including a `val_auc` alias value—come from each record's capability map. Legacy and malformed metadata states remain distinct and malformed records cannot be compared.
- Prediction: valid legacy is selectable and classification-compatible; malformed metadata is unavailable. Single/batch regression diagnostics and continuous values remain intact, classification summaries/columns remain intact, and model/JSON/CSV changes invalidate stale results. Two deferred file reads and a file-change/late-response race are covered.
- Accessibility: workflow and prediction model selectors retain explicit labels; refresh/JSON/CSV controls and live/status result regions remain accessible.

## Review round 1 — concerns

- No blocker. Build output contains only the existing third-party `@vueuse/core` PURE-annotation notices; build succeeded.
