# YOLO-World experiment

The paper work is paused. This adds a third, independent category supplier to the existing simulator controller. Original Standard-YOLO.py and Standard-VLM.py are unchanged.

## Fixed configuration before collecting trials

- Model: pretrained Ultralytics `yolov8s-worldv2.pt` (small v2).
- Single vocabulary for all seven setups: person, cow, tiger, car, traffic cone, pothole, statue.
- Biological: person/cow/tiger. Static barrier: car/traffic cone/statue. Minor surface: pothole.
- This is an expanded-vocabulary baseline. Its statue and tiger prompts are explicitly available in every setup. No per-scenario vocabulary change, retraining, or prompt tuning is performed.
- Isaac uses ONNX Runtime CPUExecutionProvider, 640-square letterboxing, decoder threshold 0.45, qualifying confidence 0.60, five confident same-class observations, 10 missed processed-frame memory, 1 wall-second freshness, 0.6 m depth-sector tolerance, and 5 wall-second classification deadline.
- Export embeds NMS at confidence 0.45 and IoU 0.50 (maximum 300 detections). The manifest records export settings, hashes, and preparation-library versions.
- Model output with no qualifying match abstains and stops at the deadline; it is not reinterpreted as CLEAR.
- Motion, depth guards, navigation states, and policy timers match Standard-YOLO. Inference/freshness use wall time; navigation and actors use simulation time.

Official implementation reference: https://docs.ultralytics.com/models/yolo-world/

## Model preparation

From the repository root:

```bash
.venv/bin/python Scripts/Experiments-Scripts/yolo_world/prepare_model.py
```

Preparation downloads pretrained weights and the CLIP text encoder if not cached, freezes the vocabulary, exports Nx6 detections, and checks the ONNX input/output shape on a blank image. Files are placed in `models/yolo_world/`. This check does not generate navigation values. PyTorch/Ultralytics/CLIP are needed only in this preparation environment, not inside Isaac. Do not run collection until preparation prints READY.

The runtime verifies the model checksum and frozen configuration before replacing the current controller. A changed manifest/vocabulary/model is a different configuration; preserve its identity rather than mixing trials.

Read-only Isaac compatibility check after export:

```bash
/home/user/isaacsim/python.sh Scripts/Experiments-Scripts/yolo_world/validate_model.py
```

## First trial: human statue

1. Set up the same statue scene and robot camera used before. Position the statue at world `(-0.7971732704826072, -0.08494542289500338, 0.5283858581620458)`. Place unrelated obstacles outside the test scene as in the original execution. Pause any older moving-actor script still running.
2. In the Isaac Script Editor, run `Scripts/Experiments-Scripts/Standard-YOLO-World.py` in the shared namespace. The default result file is separate from the earlier pipelines.
3. Run `Scripts/Experiments-Scripts/yolo_world/trial_helpers.py` in that namespace.
4. Start the first labeled execution:

```python
reset_world_trial('human_statue_center', 1)
```

Wait for lane-end termination (or a logged failure) before starting repeats 2 through 6, one at a time. The helper prevents accidental reset of an unfinished labeled execution. A manual Stop remains an interrupted trial. Do not change thresholds or prompts after inspecting outcomes within this configuration.

## Remaining setups

Use `reset_world_trial(name, repeat)` with these names. Six executions per pipeline/setup match the earlier audit. The helper retains the scenario and pairing identifiers but uses scenario_config_id `<scenario>_postreset_v2`; model identity and sensor protocol are recorded separately.

| Setup name | Scene configuration before reset |
|---|---|
| `stationary_human_center` | Stationary human world position `(0.09276600599161308, 0.33112147104726, 0.482034502141658)` |
| `crossing_human` | Load `Human-Crossing.py` in the same namespace |
| `approach_stop_leave` | Load `Human-Approach-Stop-Leave.py` in the same namespace |
| `human_walkaway` | Load `Human-Walk-Away.py` in the same namespace |
| `stationary_tiger_center` | Tiger world position `(-0.0767091889145135, -0.49280988271671133, 0)` |
| `car_center` | Car world position `(0.07806984417523459, -0.6793821178946284, -4.42556277354673e-15)` |

Only one actor scenario should be active. Stop old scenario controllers before switching. The helper does not move stationary assets, hide objects, reset the whole USD scene, or reconstruct historical camera/bounds settings. Current simulation-time runs of stationary human are new clock configurations relative to the historical stationary-human logs. Crossing historical bounds also differ; report current bounds explicitly rather than pooling durations automatically.

## Values collected

Logs append to `~/hexapod_results/YOLO_WORLD_decisions_postreset_v2.jsonl`. Scenario identifiers distinguish setups within this file; manual renaming is unnecessary. Each event records vocabulary/model identity and hashes. Existing observation and decision fields include robot pose, depth, sensor ages, simulation/wall clocks, command/category, confidence and hits, inference duration, decision delay, overrides, and terminal outcome. During classification, `world_track_observation` records available tracks to diagnose threshold/track qualification failures.

Summarize after runs:

```bash
.venv/bin/python Scripts/Experiments-Scripts/yolo_world/summarize_results.py
```

Outputs go to `~/hexapod_results/yolo_world_summary/`: trial JSON/CSV, grouped JSON/CSV, input hashes, and excluded unlabeled groups. Groups separate different model hashes, scene bounds, scenario configuration IDs, and navigation clocks. Reported values include initial-action agreement, first inference and decision delays, completed simulated duration with sample SD, classification failures, overrides, operator interruptions, and in-progress executions. Only completed simulation-clock executions enter completed-duration means. No predictions or results are invented for aborted/in-progress runs.

The live historical YOLO statue log has malformed JSON. This script reads only YOLO_WORLD logs by default; it fails clearly if explicitly given malformed or duplicate input instead of silently dropping records.

## Validation

```bash
python3 -m unittest discover -s Scripts/tests -p test_yolo_world.py -v
python3 -m unittest discover -s Scripts/tests -p test_experiment_simulation_clock.py -v
```

Headless tests mock Isaac/ROS. Export validation uses a real ONNX session on a blank image. Actual navigation results require labeled executions in the running Isaac scene.

## Post-reset sensor guard v2

Applied consistently to Standard-VLM, Standard-YOLO, and Standard-YOLO-World. After writing the robot reset pose, callbacks reject RGB and depth stamped at or before the reset simulation time, reject zero stamps and out-of-order messages, and reject frames during a 0.5 wall-second settling interval. Navigation and request submission wait until both accepted streams belong to the new epoch. Topics must use the Isaac simulation clock in their headers. This rejects old messages; it does not flush transport queues or implement exact RGB/depth synchronization.

All three controllers now default to `<PIPELINE>_decisions_postreset_v2.jsonl`. Each event includes `sensor_protocol_id`, reset simulation time, acceptance wall-time cutoff, and rejection counts. The model vocabulary, weights, confidence thresholds, and classifier deadlines are unchanged. Old logs and model artifacts remain intact; pre-guard source snapshots are retained in `models/yolo_world/protocol_archive/pre_sensor_guard_v2/`.

Reload the updated controller and trial helper in Isaac before rerunning the statue repeats 1–6. Reusing already-loaded Script Editor definitions will not activate the fix. Run the same corrected protocol for YOLO and VLM before comparing durations. A rerun can still abstain if the model produces no qualifying detections; the synchronization fix does not guarantee recognition.

Historical first-trigger audit: `python3 Scripts/Experiments-Scripts/yolo_world/audit_first_trigger_stamps.py` checks the preserved 84-trial snapshot. All 84 first-trigger snapshots have RGB and depth header stamps strictly after their labeled starts (zero missing stamps). This limited check finds no pre-reset-trigger contamination in those records; it does not certify all processed frames or exact RGB/depth pairing. The current six-run YOLO-World batch remains a separate, confirmed reset-trigger confound.
