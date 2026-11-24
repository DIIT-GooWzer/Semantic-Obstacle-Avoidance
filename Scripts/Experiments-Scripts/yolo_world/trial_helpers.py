"""Isaac Script Editor helper, run after Standard-YOLO-World.py.
For moving scenes, also run the corresponding Human-*.py script first.
Example: reset_world_trial('human_statue_center', 1)
This helper does not reposition stationary assets or hide other obstacles.
"""
WORLD_SCENARIOS = {
    'stationary_human_center': ('perform_full_reset', 'human_center', 'yield_and_wait'),
    'crossing_human': ('reset_crossing_trial', 'crossing_human', 'yield_and_wait'),
    'approach_stop_leave': ('reset_approach_trial', 'approach_stop_leave', 'yield_and_wait'),
    'human_walkaway': ('reset_walkaway_trial', 'human_walkaway', 'yield_and_wait'),
    'stationary_tiger_center': ('perform_full_reset', 'tiger_center', 'yield_and_wait'),
    'human_statue_center': ('perform_full_reset', 'human_statue', 'standard_bypass'),
    'car_center': ('perform_full_reset', 'car_center', 'standard_bypass'),
}

def reset_world_trial(scenario_id, repeat):
    """Start a labeled repetition; a trial already running must finish first."""
    if globals().get('PIPELINE') != 'YOLO_WORLD':
        raise RuntimeError('Run Standard-YOLO-World.py in this namespace first')
    if scenario_id not in WORLD_SCENARIOS:
        raise ValueError('Choose a setup from WORLD_SCENARIOS')
    if isinstance(repeat, bool) or not isinstance(repeat, int) or not 1 <= repeat <= 6:
        raise ValueError('repeat must be an integer from 1 to 6')
    # The initial unlabeled startup can be reset. Labeled executions must end
    # naturally or be stopped explicitly, to avoid accidental reset interruptions.
    context = globals().get('_trial_context', {})
    if context.get('scenario_id') and not globals().get('_trial_finished', False):
        raise RuntimeError('The labeled trial is still running; wait for lane end or explicitly Stop it')
    callback_name, instance_prefix, expected = WORLD_SCENARIOS[scenario_id]
    callback = globals().get(callback_name)
    if not callable(callback):
        raise RuntimeError('Load the matching Human scenario script first: ' + callback_name)
    return callback({
        'scenario_id': scenario_id,
        'scenario_config_id': scenario_id + '_postreset_v2',
        'scenario_instance_id': f'{instance_prefix}_repeat{repeat:02d}',
        'expected_initial_command': expected,
    })
