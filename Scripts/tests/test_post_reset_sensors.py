"""Regression tests of the actual callbacks and request/motion gates, no live Isaac."""
import contextlib
import io
import time
import types
import unittest
from unittest.mock import patch
import numpy as np
from test_pipeline_isolation import load_controller
import test_yolo_world as world_fixtures

PIPELINES=('VLM','YOLO','YOLO-World')

class PostResetSensorTests(unittest.TestCase):
    def setUp(self):
        self.fixture=world_fixtures.WorldTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def load(self, pipeline):
        if pipeline=='YOLO-World':env=self.fixture.load()
        else:
            env=load_controller('Experiments-Scripts/Standard-'+pipeline+'.py')
            self.fixture.envs.append(env) if '_worker_stop' in env else None
        env['ros_img_to_numpy']=lambda msg,desired_encoding='passthrough':msg.array
        env['timeline'].get_current_time.return_value=20.0
        with patch.object(time,'monotonic',return_value=100.0):env['_ros_node'].clear_frames()
        return env

    def msg(self,stamp,depth=False,value=.7):
        sec=int(stamp);ns=round((stamp-sec)*1e9)
        shape=(10,12) if depth else (10,12,3)
        return types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=sec,nanosec=ns)),
            array=np.full(shape,value,dtype=np.float32))

    def feed(self,node,stamp,wall=100.6,rgb=True,depth=True):
        with patch.object(time,'monotonic',return_value=wall):
            if rgb:node._rgb_callback(self.msg(stamp))
            if depth:node._depth_callback(self.msg(stamp,depth=True))

    def test_delayed_pre_reset_frames_rejected_even_after_settling(self):
        for p in PIPELINES:
            env=self.load(p);n=env['_ros_node'];seq=n.rgb_sequence
            for stamp in (0,19.9,20.0):self.feed(n,stamp,wall=101.0)
            self.assertEqual(n.rgb_sequence,seq,p)
            self.assertIsNone(n.latest_rgb,p);self.assertIsNone(n.latest_depth,p)
            self.assertEqual(n.rejected_sensor_frames,{'rgb':3,'depth':3})

    def test_settling_drops_post_reset_frames_and_requires_both_streams(self):
        for p in PIPELINES:
            env=self.load(p);n=env['_ros_node']
            self.feed(n,20.1,wall=100.49)
            self.assertIsNone(n.latest_rgb,p)
            self.feed(n,20.2,rgb=True,depth=False)
            with patch.object(time,'monotonic',return_value=100.6):self.assertFalse(n.sensors_ready(),p)
            self.feed(n,20.2,rgb=False,depth=True)
            with patch.object(time,'monotonic',return_value=100.6):self.assertTrue(n.sensors_ready(),p)
            with patch.object(time,'monotonic',return_value=101.7):self.assertFalse(n.sensors_ready(),p)

    def test_out_of_order_and_duplicate_frames_do_not_replace_newer_data(self):
        for p in PIPELINES:
            env=self.load(p);n=env['_ros_node'];self.feed(n,20.3);seq=n.rgb_sequence
            self.feed(n,20.2,wall=100.7);self.feed(n,20.3,wall=100.8)
            self.assertEqual(n.rgb_sequence,seq,p)
            self.assertEqual(n._rgb_stamp,(20,300000000),p)
            self.assertAlmostEqual(n.depth_received_at,100.6)

    def test_motion_and_classification_wait_for_new_epoch(self):
        for p in PIPELINES:
            env=self.load(p);n=env['_ros_node'];self.feed(n,19.9)
            env['drain_model_results']=lambda:None
            env['current_pos']=[0.,0.,.1];env['step_start_pos']=[0.,0.,.1]
            env['last_time']=20.;env['timeline'].get_current_time.return_value=20.1
            with patch.object(time,'monotonic',return_value=100.6):
                env['zig_zag_update'](None);env['begin_classification'](.7)
            self.assertEqual(env['current_pos'],[0.,0.,.1],p)
            self.assertEqual(env['state'],'MOVING',p)
            self.assertIsNone(env['_active_request'],p)
            self.assertIn('post-reset',env['status_label'].text)

    def test_yolo_submission_rejects_cached_pre_epoch_receipts(self):
        for p in ('YOLO','YOLO-World'):
            env=self.load(p);n=env['_ros_node'];self.feed(n,20.1)
            # Even a post-reset header cannot make an older cached receipt valid.
            n.rgb_received_at=100.4
            with patch.object(time,'monotonic',return_value=100.6):env['submit_latest_frame']()
            self.assertEqual(env['_last_submitted_sequence'],-1,p)

    def test_reset_arms_sensor_floor_after_writing_pose_and_logs_protocol(self):
        for p in PIPELINES:
            env=self.load(p);order=[];node=env['_ros_node'];clear=node.clear_frames
            env['update_usd_transform']=lambda *args:order.append('pose')
            def arm():
                order.append('arm');clear()
            node.clear_frames=arm
            with patch.object(time,'monotonic',return_value=200.):env['perform_full_reset']()
            self.assertEqual(order,['pose','arm'],p)
            self.assertIsNone(node._rgb_stamp,p)
            self.assertEqual(node.sensor_reset_sim_ns,20_000_000_000,p)
            row=env['log_decision']('test')
            self.assertEqual(row['sensor_protocol_id'],'post_reset_stamp_guard_v2')
            self.assertEqual(row['sensor_settle_wall_s'],.5)

    def test_callbacks_and_readiness_are_identical_across_all_pipelines(self):
        import ast
        from test_pipeline_isolation import SCRIPTS
        classes=[]
        for p in PIPELINES:
            tree=ast.parse((SCRIPTS/('Experiments-Scripts/Standard-'+p+'.py')).read_text())
            classes.append(ast.dump(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PerceptionNode'),include_attributes=False))
        self.assertEqual(classes[0],classes[1]);self.assertEqual(classes[1],classes[2])

if __name__=='__main__':unittest.main()
