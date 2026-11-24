import json, glob, os, random, shutil
import numpy as np
from PIL import Image

SRC = "/home/user/Desktop/llmforhexapod/dataset/replicator_out"
DST = "/home/user/Desktop/llmforhexapod/dataset/yolo_dataset"
CLASSES = {"cow": 0, "person": 1, "cone": 2, "car": 3, "pothole": 4}  # must match YOLO_CLASS_NAMES in your sim script
MAX_OCCLUSION = 0.7

for split in ("train", "val"):
    os.makedirs(f"{DST}/images/{split}", exist_ok=True)
    os.makedirs(f"{DST}/labels/{split}", exist_ok=True)

frames = sorted(glob.glob(f"{SRC}/rgb_*.png"))
print(f"Found {len(frames)} candidate images")

random.seed(0)
random.shuffle(frames)
n_val = max(1, int(len(frames) * 0.15))

converted = 0
skipped_no_files = 0
skipped_no_boxes = 0

for idx, img_path in enumerate(frames):
    fid = os.path.basename(img_path)[4:-4]            # "rgb_0000.png" -> "0000"
    npy = f"{SRC}/bounding_box_2d_tight_{fid}.npy"
    lab = f"{SRC}/bounding_box_2d_tight_labels_{fid}.json"

    if not (os.path.exists(npy) and os.path.exists(lab)):
        skipped_no_files += 1
        continue

    boxes = np.load(npy, allow_pickle=True)
    id2name = {int(k): v["class"] for k, v in json.load(open(lab)).items()}
    W, H = Image.open(img_path).size

    lines = []
    for b in boxes:
        name = id2name.get(int(b["semanticId"]))
        if name not in CLASSES:
            continue
        if float(b["occlusionRatio"]) > MAX_OCCLUSION:
            continue

        x1, y1, x2, y2 = float(b["x_min"]), float(b["y_min"]), float(b["x_max"]), float(b["y_max"])
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue

        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H
        lines.append(f"{CLASSES[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if not lines:
        skipped_no_boxes += 1
        continue

    split = "val" if idx < n_val else "train"
    shutil.copy(img_path, f"{DST}/images/{split}/{fid}.png")
    with open(f"{DST}/labels/{split}/{fid}.txt", "w") as f:
        f.write("\n".join(lines))
    converted += 1

with open(f"{DST}/custom.yaml", "w") as f:
    f.write(f"""path: {DST}
train: images/train
val: images/val
names:
  0: cow
  1: person
  2: cone
  3: car
  4: pothole
""")

print(f"Converted: {converted}")
print(f"Skipped (missing npy/json): {skipped_no_files}")
print(f"Skipped (no valid boxes after filters): {skipped_no_boxes}")
print(f"Done. Dataset written to {DST}")
