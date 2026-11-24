"""World adapter/provenance and result aggregation checks; Isaac is mocked."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from test_pipeline_isolation import load_controller, SCRIPTS

WORLD = SCRIPTS/'Experiments-Scripts/Standard-YOLO-World.py'
CFG = SCRIPTS/'Experiments-Scripts/yolo_world/vocabulary_v1.json'

class WorldTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.envs = []
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        root = Path(self.folder.name)
        config = CFG.read_bytes()
        self.cfg = json.loads(config)
        (root/'model.onnx').write_bytes(b'test fixture, not a trained model')
        (root/'vocabulary_v1.json').write_bytes(config)
        manifest = {'config': self.cfg, 'onnx_file':'model.onnx',
                    'onnx_sha256':hashlib.sha256((root/'model.onnx').read_bytes()).hexdigest(),
                    'config_sha256':hashlib.sha256(config).hexdigest(),
                    'checkpoint':'test-fixture.pt', 'checkpoint_sha256':'fixture',
                    'export':{'nms':True}}
        (root/'manifest.json').write_text(json.dumps(manifest))

    def tearDown(self):
        for env in self.envs:
            env['_worker_stop'].set()
            env['_inference_thread'].join(timeout=1)
        self.folder.cleanup()
        self.output.__exit__(None, None, None)

    def load(self, names=None):
        if names is None:
            names = {i:x['name'] for i,x in enumerate(self.cfg['classes'])}
        with patch.dict(os.environ, {'PHANTOMX_YOLO_WORLD_DIR':self.folder.name}):
            env = load_controller('Experiments-Scripts/Standard-YOLO-World.py', onnx_names=names)
        self.envs.append(env)
        return env

    def test_full_startup_logs_world_identity_without_ultralytics_in_isaac(self):
        env = self.load()
        row = env['log_decision']('test')
        self.assertEqual(row['pipeline'], 'YOLO_WORLD')
        self.assertEqual(len(row['pipeline_metadata']['vocabulary']), 7)
        self.assertEqual(env['YOLO_CLASS_TO_CATEGORY']['statue'], 'STATIC_BAR')
        self.assertNotIn('ultralytics', [n.names[0].name for n in ast.walk(ast.parse(WORLD.read_text())) if isinstance(n,ast.Import)])

    def test_model_tampering_stops_before_startup(self):
        (Path(self.folder.name)/'model.onnx').write_bytes(b'wrong model')
        with self.assertRaisesRegex(RuntimeError, 'hash differs'):
            self.load()

    def test_vocabulary_tampering_is_rejected(self):
        path=Path(self.folder.name)/'vocabulary_v1.json'
        path.write_text(path.read_text().replace('statue','mannequin'))
        with self.assertRaisesRegex(RuntimeError, 'vocabulary differs'):
            self.load()

    def test_onnx_name_order_must_match_manifest(self):
        with self.assertRaisesRegex(ValueError, 'vocabulary differs'):
            self.load({0:'wrong'})

    def test_five_statue_observations_select_shared_bypass(self):
        env = self.load()
        for sequence in range(1,6):
            env['update_tracks'](sequence,time.monotonic(),(2,.7,2),
                [{'bbox':[40,20,60,80],'confidence':.9,'class_name':'statue'}],.04)
        self.assertIsNotNone(env['find_confident_yolo_match'](.7))
        env['begin_classification'](.7)
        self.assertEqual(env['active_obstacle_category'],'STATIC_BAR')
        self.assertEqual(env['state'],'AVOID_INITIAL_TURN')

    def test_no_detection_abstains_instead_of_clear(self):
        env = self.load()
        env['state']='CLASSIFYING'; env['_active_request']=1
        env['classify_started_at']=time.monotonic()-6
        records=[]
        env['log_decision']=lambda status,**fields:records.append((status,fields))
        env['poll_classification']()
        self.assertEqual(env['state'],'STOPPED')
        self.assertIn('abstain',[r[0] for r in records])
        self.assertNotIn('predicted',[r[0] for r in records])

    def test_common_navigation_tracking_and_clock_code_is_identical(self):
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        base=functions(SCRIPTS/'Experiments-Scripts/Standard-YOLO.py'); world=functions(WORLD)
        allowed={'log_decision','start_inference_worker','drain_model_results'}
        for name in base.keys()-allowed:
            self.assertEqual(base[name],world[name],name)
        self.assertGreater(len(base.keys()-allowed),30)

class SummaryTests(unittest.TestCase):
    def test_aborted_and_in_progress_are_not_completed_durations(self):
        path=SCRIPTS/'Experiments-Scripts/yolo_world/summarize_results.py'
        spec=importlib.util.spec_from_file_location('world_summary',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/'fixture.jsonl';events=[]
            seq=0
            for trial,outcome in [('a','completed'),('b','aborted'),('c',None)]:
                for status in ['trial_started','predicted']+(['run_finished'] if outcome else []):
                    seq+=1
                    e={'session_id':'session','trial_id':trial,'event_sequence':seq,
                       'pipeline':'YOLO_WORLD','scenario_id':'statue','scenario_config_id':'v1',
                       'navigation_clock':'simulation_time','status':status,
                       'sim_time':0 if status=='trial_started' else 12,
                       'monotonic_time':0 if status=='trial_started' else 24,
                       'expected_initial_command':'standard_bypass','command':'standard_bypass',
                       'category':'STATIC_BAR','decision_latency_s':.1,'inference_s':.04}
                    if status=='run_finished':e.update(outcome=outcome,reason='reset_requested' if outcome=='aborted' else 'Lane End Reached')
                    events.append(e)
            f.write_text('\n'.join(json.dumps(e) for e in events))
            trials,summary,_,_=module.analyze([f]);s=summary[0]
            self.assertEqual((s['completed'],s['aborted'],s['in_progress']),(1,1,1))
            self.assertEqual(s['completed_sim_duration_n'],1)
            self.assertEqual(s['operator_interrupted'],1)
            self.assertEqual(s['completed_sim_duration_mean_s'],12)
            with self.assertRaisesRegex(ValueError,'Duplicate event'):module.analyze([f,f])

if __name__=='__main__':unittest.main()
