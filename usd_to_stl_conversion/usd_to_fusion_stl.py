#!/usr/bin/env python3
"""
Convert visual USD mesh prims to STL while preserving PhantomX scale for Fusion.

Recommended for this asset:
  python usd_to_fusion_stl.py phantomX_base.usd --root /visuals/tibia_LF --scale 1000 --out tibia_LF_mm.stl
  python usd_to_fusion_stl.py phantomX_base.usd --root /visuals --split-children --scale 1000 --out-dir fusion_stl_mm

Why scale=1000?
The USD stage is authored in meters (metersPerUnit=1). Fusion/STL workflows usually expect mm.
STL is unitless, so this writes vertex numbers in millimeters.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, Gf


def safe_name(path: str) -> str:
    s = path.strip('/').replace('/', '__') or 'root'
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)


def triangulate(face_counts: Iterable[int], face_indices: Iterable[int]) -> np.ndarray:
    indices = list(face_indices)
    faces: List[Tuple[int, int, int]] = []
    cursor = 0
    for count in face_counts:
        poly = indices[cursor:cursor + count]
        cursor += count
        if count < 3:
            continue
        if count == 3:
            faces.append((poly[0], poly[1], poly[2]))
        else:
            # Fan triangulation. Fine for imported STL-style triangulated assets; handles quads/ngons too.
            for i in range(1, count - 1):
                faces.append((poly[0], poly[i], poly[i + 1]))
    return np.asarray(faces, dtype=np.int64)


def get_world_matrix(xform_cache: UsdGeom.XformCache, prim: Usd.Prim) -> np.ndarray:
    m: Gf.Matrix4d = xform_cache.GetLocalToWorldTransform(prim)
    return np.asarray(m, dtype=np.float64).reshape(4, 4)


def mesh_from_usd_prim(stage: Usd.Stage, prim: Usd.Prim, scale: float) -> trimesh.Trimesh | None:
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or not indices:
        return None

    verts = np.asarray([(p[0], p[1], p[2]) for p in points], dtype=np.float64)
    faces = triangulate(counts, indices)
    if len(verts) == 0 or len(faces) == 0:
        return None

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mat = get_world_matrix(xform_cache, prim)
    verts_h = np.c_[verts, np.ones(len(verts))]
    verts_world = (verts_h @ mat)[:, :3] * scale
    return trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)


def collect_meshes(stage: Usd.Stage, root_path: str) -> List[trimesh.Trimesh]:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise ValueError(f'USD root path not found: {root_path}')

    meshes: List[trimesh.Trimesh] = []
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == 'Mesh':
            tm = mesh_from_usd_prim(stage, prim, args.scale)
            if tm is not None:
                meshes.append(tm)
    return meshes


def export_one(stage: Usd.Stage, root_path: str, out_path: Path, scale: float) -> None:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise ValueError(f'USD root path not found: {root_path}')

    meshes: List[trimesh.Trimesh] = []
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == 'Mesh':
            tm = mesh_from_usd_prim(stage, prim, scale)
            if tm is not None:
                meshes.append(tm)

    if not meshes:
        raise RuntimeError(f'No Mesh prims found below {root_path}. For this asset, try --root /visuals or a child like /visuals/tibia_LF.')

    merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.export(out_path)
    bounds = merged.bounds
    size = bounds[1] - bounds[0]
    print(f'Wrote {out_path}')
    print(f'  meshes={len(meshes)} vertices={len(merged.vertices)} faces={len(merged.faces)}')
    print(f'  bbox size in exported units: X={size[0]:.6g}, Y={size[1]:.6g}, Z={size[2]:.6g}')


parser = argparse.ArgumentParser()
parser.add_argument('usd', help='Input USD file. For this PhantomX asset, use phantomX_base.usd for visual geometry.')
parser.add_argument('--root', default='/visuals', help='USD prim path to export. Examples: /visuals, /visuals/tibia_LF, /visuals/coxa_LF')
parser.add_argument('--scale', type=float, default=1000.0, help='Scale applied to final USD world vertices. Use 1000 to write meter USD as millimeter STL.')
parser.add_argument('--out', default=None, help='Output STL path for one combined export')
parser.add_argument('--split-children', action='store_true', help='Export each direct child below --root as a separate STL')
parser.add_argument('--out-dir', default='fusion_stl_mm', help='Directory for --split-children outputs')
args = parser.parse_args()

stage = Usd.Stage.Open(args.usd)
mpu = UsdGeom.GetStageMetersPerUnit(stage)
up = UsdGeom.GetStageUpAxis(stage)
print(f'Opened {args.usd}')
print(f'  metersPerUnit={mpu}, upAxis={up}, export scale={args.scale}')

root = stage.GetPrimAtPath(args.root)
if not root or not root.IsValid():
    raise SystemExit(f'Root path not found: {args.root}')

if args.split_children:
    out_dir = Path(args.out_dir)
    exported = 0
    for child in root.GetChildren():
        try:
            out = out_dir / f'{safe_name(str(child.GetPath()))}.stl'
            export_one(stage, str(child.GetPath()), out, args.scale)
            exported += 1
        except RuntimeError as e:
            print(f'Skip {child.GetPath()}: {e}')
    print(f'Exported {exported} STL file(s) into {out_dir}')
else:
    out = Path(args.out or f'{safe_name(args.root)}.stl')
    export_one(stage, args.root, out, args.scale)
