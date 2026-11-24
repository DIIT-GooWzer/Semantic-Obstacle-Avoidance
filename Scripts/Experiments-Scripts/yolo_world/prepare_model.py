"""Export YOLO-World with a fixed vocabulary; no simulator or trial collection.
Run from repository root: .venv/bin/python Scripts/Experiments-Scripts/yolo_world/prepare_model.py
Only model preparation uses PyTorch/CLIP. Isaac uses the exported ONNX artifact.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name('vocabulary_v1.json')
OUT = ROOT / 'models' / 'yolo_world'

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLOWorld
    import onnxruntime as ort

    cfg = json.loads(CONFIG.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / cfg['checkpoint']
    model = YOLOWorld(str(checkpoint))
    names = [row['name'] for row in cfg['classes']]
    model.set_classes(names)
    # Freeze text embeddings into the ONNX model. Match the existing Nx6 decoder.
    started = time.monotonic()
    exported = Path(model.export(format='onnx', imgsz=cfg['image_size'],
                                 batch=1, dynamic=False, simplify=False,
                                 nms=True, conf=cfg['decode_confidence'],
                                 iou=cfg['nms_iou'], max_det=300,
                                 device=cfg['device'], opset=17))
    session = ort.InferenceSession(str(exported), providers=['CPUExecutionProvider'])
    shape = session.get_inputs()[0].shape
    assert shape == [1, 3, 640, 640], shape
    sample = np.zeros(shape, dtype=np.float32)
    # Warm-up + schema smoke check only. This is not an experiment trial.
    result = session.run(None, {session.get_inputs()[0].name: sample})[0]
    if result.ndim != 3 or result.shape[0] != 1 or result.shape[-1] != 6:
        raise RuntimeError(f'Unsupported exported detections: {result.shape}')
    embedded_names = session.get_modelmeta().custom_metadata_map.get('names')
    import ast
    decoded_names = ast.literal_eval(embedded_names)
    assert [decoded_names[i] for i in range(len(names))] == names
    manifest = {
        'config_id': cfg['config_id'], 'config': cfg,
        'config_sha256': sha256(CONFIG),
        'checkpoint': checkpoint.name, 'checkpoint_sha256': sha256(checkpoint),
        'onnx_file': exported.name, 'onnx_sha256': sha256(exported),
        'ultralytics_version': ultralytics.__version__,
        'torch_version': torch.__version__, 'onnxruntime_version': ort.__version__,
        'python_version': sys.version, 'platform': platform.platform(),
        'clip_version': importlib.metadata.version('clip'),
        'clip_install_source': json.loads(importlib.metadata.distribution('clip').read_text('direct_url.json') or '{}'),
        'input_shape': shape, 'output_shape': list(result.shape),
        'preparation_wall_s': time.monotonic() - started,
        'validation': 'zero-image schema smoke check; not a navigation trial',
        'export': {'format': 'onnx', 'opset': 17, 'nms': True, 'conf': cfg['decode_confidence'],
                   'iou': cfg['nms_iou'], 'max_det': 300, 'dynamic': False, 'simplify': False},
    }
    (OUT / 'vocabulary_v1.json').write_bytes(CONFIG.read_bytes())
    manifest_path = OUT / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('READY:', exported)
    print('Manifest:', manifest_path)
    print('No navigation experiment values have been collected by this preparation step.')

if __name__ == '__main__':
    main()
