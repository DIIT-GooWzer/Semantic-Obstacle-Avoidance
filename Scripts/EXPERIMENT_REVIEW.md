# Isolated obstacle-decision pipelines

The scripts are now standalone, mutually exclusive pipelines:

| File | Classification model | Dependencies |
| --- | --- | --- |
| `amazing8.py` | VLM HTTP request only | OpenCV and the configured VLM server; no ONNX runtime |
| `amazing8-novlm.py` | YOLO ONNX only | OpenCV and ONNX Runtime; no HTTP/VLM client |

Both retain the existing kinematic navigation, bypass routines, ROS RGB/depth
input, and depth-based emergency guards. Depth guards are common navigation
controls, not another semantic classifier. No cross-model refinement, stress or
trauma overrides, obstacle-category cache, or automatic dataset capture remains.
The one legacy `yolo_thread_running = False` cleanup assignment stops workers from
older Script Editor runs; it never starts inference.

Both stop at the safe Y boundary in the current lane's travel direction, logging
`Lane End Reached` as completed. They do not turn into another lane. Maneuvers
that formerly skipped a blocked lane now stop with `Lane blocked` as aborted;
legacy shift/skip states are also guarded against motion. Automatic repeat is
disabled, so these stops persist until a manual reset.

## Decision behavior

Both scripts use the same mapping:

| Category | Command |
| --- | --- |
| BIOLOGICAL | yield_and_wait |
| STATIC_BAR | standard_bypass |
| MINOR_SURFACE | scan_and_proceed |
| CLEAR | resume_path |

VLM responses provide a category and optional brief reason; the controller validates
that category and maps it to the command. Sector and risk outputs no longer change
maneuvers. This is category-to-action pipeline evaluation, not contextual reasoning.

The VLM script retains its configured endpoint and `qwen3.5-9b` model identifier.
Authentication uses `LM_STUDIO_API_KEY` with an empty default; configure it if your
server requires authentication. The script does not silently select another model.

The YOLO script retains the ONNX path, class names, 0.45 detection threshold,
0.60 acceptance threshold, five-hit requirement, and 0.6 m depth matching tolerance.
All five hits must meet the acceptance threshold and come from distinct processed
camera frames. They may now span bounded detection gaps: `TRACK_MAX_AGE = 10`
retains unmatched identities for up to ten missed processed frames, also limited
to `SENSOR_MAX_AGE` (one second) since their last actual detection. Misses preserve
the hit count without increasing it; a below-acceptance detection or a class
change resets that count. This is a gap-tolerant observation rule, not five
consecutive unbroken frames.

Carried tracks are association memory only and cannot trigger a command. A fresh
matching detection is required before a preserved count becomes actionable.
Duplicate/out-of-order results neither increment hits nor age tracks. Prediction
metadata records the persistence limit and confidence-rule name. These settings
must be frozen before collecting comparison runs and validated on motion cases;
adding persistence alone does not establish comparative fairness or accuracy.

Matching is one-to-one per frame. Only end-to-end `(1,N,6)` output
is accepted; incompatible exports produce an explicit error rather than guessed
predictions. The model is required at startup.

Both controllers stop on classification failure and log an abstention. There is
no default STATIC_BAR prediction or bypass presented as a model decision. YOLO
waits up to `CLASSIFICATION_WAIT_SECONDS` (5 s); VLM uses `VLM_TIMEOUT` (360 s).
These deadlines differ and should be set deliberately before experimental runs.

Workers only perform inference; the simulation thread applies results. Request
identifiers invalidate old VLM responses after stop/reset. YOLO uses a generation
identifier to discard results from before a reset. Reset clears camera buffers
and waits for new sensor data. Missing/stale RGB or depth holds navigation.

## Logging and operation

The UI shows selected pipeline, predicted category/command, depth, and navigation
status. Logs default to `~/hexapod_results/VLM_decisions.jsonl` and
`~/hexapod_results/YOLO_decisions.jsonl` and include predictions, abstentions,
inference duration, trigger-to-decision latency, and finish reasons. YOLO's
trigger-to-decision latency may use tracks acquired while approaching; it is not
interchangeable with per-frame inference duration.

The initialization-order bug involving `MIN_DIVERGE_STEP` is fixed. Emergency
escape yaw selection also returns the scalar heading expected by its caller.
Motion/yield timing now uses a monotonic wall clock, still not simulation time.
Yielding retains the shared existing wait-then-bypass behavior after ten seconds.

Automatic repeating is disabled in both scripts. Both now select the first valid
NavMesh from the same ordered list: `/World/Terrain/section_00_asphalt/NavMeshVolume`,
then `/World/NavMeshVolume`. The selected path and bounds are printed and included
in every event. Missing, nonfinite, or too-small bounds cause an explicit setup
error; the previous guessed boundary fallback has been removed.

This change does not implement a finite scenario manifest, actor trajectories,
ground-truth labels, collision measurement, or statistical reporting. Those remain
separate experiment work. Motion remains kinematic.

## Phase 1 instrumentation

Every JSONL row has the same top-level keys across both pipelines (schema version
1). Optional measurements/outputs are null; model-specific details are under
`model_metadata`, and event-specific details are under `event_metadata`.
Nonfinite depths are represented as null, so all output is strict JSON.

- `session_id`: UUID generated at script load; `event_sequence` increases within
  that session. Rerunning a script creates a new session.
- `trial_id`: UUID for one execution; reset closes the prior unfinished execution
  as aborted and creates a new trial. Encounter IDs increase across resets.
- `scenario_id`, `scenario_config_id`, `scenario_instance_id`, and
  `expected_initial_command`: optional runner-supplied labels. They are null for
  unconfigured manual runs. Expected commands are never supplied to either model.
- `robot_pos`, `robot_yaw`, `obstacle_distance_m`, `sector_depths_m`, RGB frame
  sequence, source ROS stamps, and RGB/depth ages describe the event-time state.
- `wall_time`, `monotonic_time`, and `sim_time` identify the event's clocks. Sensor
  age uses local monotonic receipt time, not camera exposure time; ROS source
  stamps are logged separately without assuming they share the wall clock.
- `classification_start` is a frozen copy of that observation when the encounter
  triggers. It stays unchanged in prediction, abstention, override and terminal
  events for that encounter. The VLM uses the image copied at that trigger;
  YOLO also logs the contributing track frame sequence and receipt time/age.

The event sequence is `trial_started`, then zero or more `detected` followed by
`predicted` or `abstain`, and finally `run_finished`. A trial with no detection
still has start/end records. A manual stop during classification first logs an
abstention, then an aborted finish. Classification failure also closes the trial.
Repeated stop clicks do not produce duplicate terminal events.

When biological yielding times out, a `policy_override` event records the original
yield command and subsequent bypass command. The predicted BIOLOGICAL category
is preserved; timeout is not recorded as a new model classification. Decision
accuracy should use the `predicted` event, and execution analysis should include
subsequent overrides. A log write failure stops execution with an error rather
than silently continuing without evidence.

To label a manual trial now, or from the future runner, call this in the
controller's Script Editor execution namespace:

```python
perform_full_reset({
    "scenario_id": "stationary_human_center",
    "scenario_config_id": "stationary_human_center_v1",
    "scenario_instance_id": "human_center_seed42_repeat01",
    "expected_initial_command": "yield_and_wait",
})
```

Use the same `scenario_instance_id` for the corresponding execution in the other
pipeline. It is the pairing key; each execution keeps its own unique `trial_id`.
An unlabeled UI Reset retains the scenario/configuration/expected-command fields
but clears the pairing key, so a new repeat cannot silently inherit an old pair.
`trial_started` records after reset have no encounter ID until detection occurs.

Phase 1 only records simulation time. Navigation and yield timers still use
monotonic wall time. Convert those timers to simulation time before adding
simulation-time actor trajectories. Classification service deadlines and latency
measurements should continue to use monotonic wall time.

## Verification

Run `python3 -m unittest discover -s Scripts/tests -v` for mocked headless regression
checks. These execute module initialization, navigation branches, category mapping,
VLM transport success/failure, timeout/abstention behavior, stale-result filtering,
frame deduplication, confidence tracking, and reset behavior. They also check model
dependency exclusivity and matching navigation implementations.

Isaac Sim, real ROS camera streams, actual ONNX outputs, and the live VLM server
must still be checked in the target environment. Mocked tests are not an Isaac Sim
integration test.
