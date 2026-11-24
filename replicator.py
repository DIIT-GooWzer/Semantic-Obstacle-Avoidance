import asyncio
import omni.replicator.core as rep
import omni.usd
from pxr import UsdGeom

# ---- CONFIG ----
CAM_PATH   = "/World/phantomX_base/Camera"     # <-- EDIT to your real camera path
OUTPUT_DIR = "/home/user/Desktop/llmforhexapod/dataset/replicator_out"
RESOLUTION = (640, 480)
NUM_FRAMES = 100

# ---- Verify camera exists (sync, safe at module level) ----
stage = omni.usd.get_context().get_stage()
cam_prim = stage.GetPrimAtPath(CAM_PATH)
if not cam_prim.IsValid():
    raise RuntimeError(f"No camera at {CAM_PATH}")
if not cam_prim.IsA(UsdGeom.Camera):
    raise RuntimeError(f"{CAM_PATH} is not a Camera ({cam_prim.GetTypeName()})")

# ---- Capture on play off: we control stepping manually ----
rep.orchestrator.set_capture_on_play(False)

# ---- Build render product + writer (sync, safe) ----
rp = rep.create.render_product(CAM_PATH, RESOLUTION)
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir=OUTPUT_DIR,
    rgb=True,
    bounding_box_2d_tight=True,
)
writer.attach([rp])
print(f"Writer attached. Output -> {OUTPUT_DIR}")

# ---- Async capture loop ----
async def generate():
    print(f"[Replicator] Starting {NUM_FRAMES} frames...")
    for i in range(NUM_FRAMES):
        await rep.orchestrator.step_async()
        if (i + 1) % 10 == 0:
            print(f"[Replicator] {i + 1}/{NUM_FRAMES} frames captured")
    print(f"[Replicator] Done. Check {OUTPUT_DIR}")

# Fire-and-forget: schedule the coroutine on the running event loop
asyncio.ensure_future(generate())
