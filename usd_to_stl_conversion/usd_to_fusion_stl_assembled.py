#!/usr/bin/env python3
"""
Export the PhantomX visual library as an assembled Fusion-safe STL.

Why this script exists:
- /visuals contains the part meshes in their local part frames.
- /phantomX/<link_name> contains the robot assembly transforms.
- Exporting /visuals directly gives the correct part sizes, but not the assembled robot layout.

Example:
  python3 usd_to_fusion_stl_assembled.py phantomX_base.usd --scale 1000 --out phantomX_full_assembled_visuals_mm.stl

Fusion import:
  Insert > Insert Mesh > Unit Type: Millimeter
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, Gf


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
            for i in range(1, count - 1):
                faces.append((poly[0], poly[i], poly[i + 1]))
    return np.asarray(faces, dtype=np.int64)


def get_matrix(cache: UsdGeom.XformCache, prim: Usd.Prim) -> np.ndarray:
    m: Gf.Matrix4d = cache.GetLocalToWorldTransform(prim)
    return np.asarray(m, dtype=np.float64).reshape(4, 4)


def export_assembled(stage: Usd.Stage, visual_root_path: str, link_root_path: str, out_path: Path, scale: float) -> None:
    visual_root = stage.GetPrimAtPath(visual_root_path)
    link_root = stage.GetPrimAtPath(link_root_path)
    if not visual_root or not visual_root.IsValid():
        raise ValueError(f'Visual root not found: {visual_root_path}')
    if not link_root or not link_root.IsValid():
        raise ValueError(f'Link root not found: {link_root_path}')

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meshes: List[trimesh.Trimesh] = []
    used_parts: List[str] = []
    skipped_parts: List[str] = []

    for visual_part in visual_root.GetChildren():
        name = visual_part.GetName()
        link_prim = stage.GetPrimAtPath(f'{link_root_path}/{name}')
        if not link_prim or not link_prim.IsValid():
            skipped_parts.append(name)
            continue

        link_mat = get_matrix(cache, link_prim)
        part_mesh_count = 0

        for prim in Usd.PrimRange(visual_part):
            if prim.GetTypeName() != 'Mesh':
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not points or not counts or not indices:
                continue

            verts = np.asarray([(p[0], p[1], p[2]) for p in points], dtype=np.float64)
            faces = triangulate(counts, indices)
            if len(verts) == 0 or len(faces) == 0:
                continue

            # USD matrices are used here in the same row-vector convention as the simple exporter:
            # local mesh -> /visuals/<part> local placement -> /phantomX/<part> assembled placement.
            visual_mesh_mat = get_matrix(cache, prim)
            assembled_mat = visual_mesh_mat @ link_mat
            verts_h = np.c_[verts, np.ones(len(verts))]
            verts_world = (verts_h @ assembled_mat)[:, :3] * scale

            meshes.append(trimesh.Trimesh(vertices=verts_world, faces=faces, process=False))
            part_mesh_count += 1

        if part_mesh_count:
            used_parts.append(name)

    if not meshes:
        raise RuntimeError(f'No mesh parts exported from {visual_root_path} using link transforms from {link_root_path}')

    merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.export(out_path)

    size = merged.bounds[1] - merged.bounds[0]
    print(f'Wrote {out_path}')
    print(f'  parts={len(used_parts)} meshes={len(meshes)} vertices={len(merged.vertices)} faces={len(merged.faces)}')
    print(f'  bbox size in exported units: X={size[0]:.6g}, Y={size[1]:.6g}, Z={size[2]:.6g}')
    print('  exported parts: ' + ', '.join(used_parts))
    if skipped_parts:
        print('  skipped parts without matching link: ' + ', '.join(skipped_parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('usd', help='Input USD file, usually phantomX_base.usd')
    parser.add_argument('--visual-root', default='/visuals', help='Path containing visual part meshes')
    parser.add_argument('--link-root', default='/phantomX', help='Path containing assembled link transforms')
    parser.add_argument('--scale', type=float, default=1000.0, help='Use 1000 to convert meter-authored USD to mm STL numbers')
    parser.add_argument('--out', default='phantomX_full_assembled_visuals_mm.stl', help='Output STL path')
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.usd)
    print(f'Opened {args.usd}')
    print(f'  metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)}, upAxis={UsdGeom.GetStageUpAxis(stage)}, export scale={args.scale}')
    export_assembled(stage, args.visual_root, args.link_root, Path(args.out), args.scale)


if __name__ == '__main__':
    main()
