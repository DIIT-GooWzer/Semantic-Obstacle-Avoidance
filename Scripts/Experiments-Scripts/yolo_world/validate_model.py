"""Read-only compatibility check; no robot, camera, or experiment result logging.
/home/user/isaacsim/python.sh Scripts/Experiments-Scripts/yolo_world/validate_model.py
"""
from pathlib import Path
import ast
import hashlib
import json
import platform
import time
import numpy as np
import onnxruntime as ort

root=Path(__file__).resolve().parents[3]
folder=root/'models/yolo_world'
manifest=json.loads((folder/'manifest.json').read_text())
path=folder/manifest['onnx_file']
assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['onnx_sha256']
cfg=manifest['config']
assert json.loads((folder/'vocabulary_v1.json').read_text())==cfg
session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
names=ast.literal_eval(session.get_modelmeta().custom_metadata_map['names'])
assert [names[i] for i in range(len(cfg['classes']))]==[r['name'] for r in cfg['classes']]
frame=np.zeros((1,3,640,640),dtype=np.float32)
started=time.monotonic()
output=session.run(None,{session.get_inputs()[0].name:frame})[0]
assert output.shape==(1,300,6),output.shape
print(json.dumps({'validation':'PASS: real ONNX blank-input compatibility check; not a trial',
                  'python':platform.python_version(), 'onnxruntime':ort.__version__,
                  'providers':session.get_providers(), 'output_shape':list(output.shape),
                  'onnx_sha256':manifest['onnx_sha256']},indent=2))
