"""Summarize labeled executions without treating interrupted trials as failures.
Default: ~/hexapod_results/YOLO_WORLD_decisions*.jsonl
Outputs trial rows, grouped metrics, exclusions, and input hashes in a separate directory.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import statistics

def avg(values):
    return statistics.mean(values) if values else None

def spread(values):
    return statistics.stdev(values) if len(values) > 1 else None

def analyze(paths):
    groups = collections.defaultdict(list)
    manifest = []
    seen = set()
    for path in paths:
        raw = path.read_bytes()
        manifest.append({'file': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        for line_number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                e = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(f'{path}:{line_number}: invalid JSON: {error}') from error
            key = (e['session_id'], e['event_sequence'])
            if key in seen:
                raise ValueError(f'Duplicate event {key}; do not supply overlapping copies of a log')
            seen.add(key)
            groups[e['session_id'], e['trial_id']].append(e)
    trials, excluded = [], []
    for (session, trial), events in groups.items():
        starts = [e for e in events if e['status'] == 'trial_started']
        if len(starts) != 1 or not starts[0].get('scenario_id'):
            excluded.append({'session_id': session, 'trial_id': trial, 'reason': 'missing unique labeled start'})
            continue
        start = starts[0]
        ends = [e for e in events if e['status'] == 'run_finished']
        if len(ends) > 1:
            raise ValueError(f'Multiple terminal events for {trial}')
        preds = [e for e in events if e['status'] == 'predicted']
        first = preds[0] if preds else {}
        end = ends[0] if ends else {}
        metadata = start.get('pipeline_metadata', {})
        cfg = metadata.get('config_id')
        # Prevent silently pooling exports or model changes under one config label.
        identity = (cfg, metadata.get('onnx_sha256'), metadata.get('config_sha256'))
        inferred = [p['inference_s'] for p in preds if p.get('inference_s') is not None]
        trials.append({
            'session_id': session, 'trial_id': trial,
            'pipeline': start['pipeline'], 'scenario_id': start['scenario_id'],
            'scenario_config_id': start.get('scenario_config_id'),
            'scenario_instance_id': start.get('scenario_instance_id'),
            'model_config_id': cfg, 'model_identity': identity,
            'sensor_protocol_id': start.get('sensor_protocol_id', 'legacy_unversioned'),
            'navmesh_bounds': start.get('navmesh_bounds_m'),
            'expected_initial_command': start.get('expected_initial_command'),
            'initial_category': first.get('category'), 'initial_command': first.get('command'),
            'initial_agreement': (first.get('command') == start.get('expected_initial_command')
                                  if first and start.get('expected_initial_command') else None),
            'initial_class_name': first.get('model_metadata', {}).get('class_name'),
            'initial_confidence': first.get('model_metadata', {}).get('confidence'),
            'first_decision_wall_s': first.get('decision_latency_s'),
            'first_inference_wall_s': first.get('inference_s'),
            'mean_encounter_inference_wall_s': avg(inferred),
            'prediction_count': len(preds),
            'abstention_count': sum(e['status'] == 'abstain' for e in events),
            'policy_override_count': sum(e['status'] == 'policy_override' for e in events),
            'outcome': end.get('outcome', 'in_progress'), 'terminal_reason': end.get('reason'),
            'navigation_clock': start.get('navigation_clock'),
            'elapsed_sim_s': end['sim_time'] - start['sim_time'] if end else None,
            'elapsed_wall_s': end['monotonic_time'] - start['monotonic_time'] if end else None,
        })
    buckets = collections.defaultdict(list)
    for t in trials:
        key = (t['pipeline'], t['scenario_config_id'], t['model_identity'],
               json.dumps(t['navmesh_bounds'], sort_keys=True), t['navigation_clock'], t['sensor_protocol_id'])
        buckets[key].append(t)
    summary = []
    for (pipeline, scenario, identity, bounds, clock, protocol), rows in buckets.items():
        completed = [t['elapsed_sim_s'] for t in rows if t['outcome'] == 'completed' and clock == 'simulation_time']
        lat = [t['first_decision_wall_s'] for t in rows if t['first_decision_wall_s'] is not None]
        inf = [t['first_inference_wall_s'] for t in rows if t['first_inference_wall_s'] is not None]
        pairing_counts = collections.Counter(t['scenario_instance_id'] for t in rows if t['scenario_instance_id'])
        summary.append({
            'pipeline': pipeline, 'scenario_config_id': scenario,
            'model_identity': identity, 'navmesh_bounds': json.loads(bounds),
            'sensor_protocol_id': protocol,
            'navigation_clock': clock, 'labeled_trials': len(rows),
            'trials_with_initial_prediction': len(lat),
            'initial_agreements': sum(t['initial_agreement'] is True for t in rows),
            'initial_disagreements': sum(t['initial_agreement'] is False for t in rows),
            'completed': sum(t['outcome'] == 'completed' for t in rows),
            'aborted': sum(t['outcome'] == 'aborted' for t in rows),
            'in_progress': sum(t['outcome'] == 'in_progress' for t in rows),
            'operator_interrupted': sum(t['terminal_reason'] in ('reset_requested', 'manual_stop') for t in rows),
            'classification_failed': sum(t['terminal_reason'] == 'classification_failed' for t in rows),
            'missing_pairing_ids': sum(not t['scenario_instance_id'] for t in rows),
            'duplicate_pairing_ids': [key for key,count in pairing_counts.items() if count > 1],
            'policy_overrides': sum(t['policy_override_count'] for t in rows),
            'first_decision_wall_mean_s': avg(lat), 'first_decision_wall_sd_s': spread(lat),
            'first_inference_wall_mean_s': avg(inf),
            'completed_sim_duration_n': len(completed),
            'completed_sim_duration_mean_s': avg(completed),
            'completed_sim_duration_sd_s': spread(completed),
        })
    return trials, summary, excluded, manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='*', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path.home()/'hexapod_results'/'yolo_world_summary')
    args = parser.parse_args()
    paths = args.logs or sorted((Path.home()/'hexapod_results').glob('YOLO_WORLD_decisions*.jsonl'))
    if not paths:
        print('No YOLO-World logs yet. Run the controller and labeled trials in Isaac Sim first.')
        return
    trials, summary, excluded, manifest = analyze(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [('trials',trials), ('summary',summary), ('excluded',excluded), ('inputs',manifest)]:
        (args.output_dir/(name+'.json')).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    for name, rows in [('trials', trials), ('summary', summary)]:
        if rows:
            with (args.output_dir/(name+'.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    for row in summary:
        print(json.dumps(row, allow_nan=False))
    print(f'Retained {len(trials)} labeled trials; excluded {len(excluded)} unlabeled groups.')
    print('Output:', args.output_dir)

if __name__ == '__main__':
    main()
