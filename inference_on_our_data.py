# infer_spatialvla.py
import os, glob, numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
import matplotlib.pyplot as plt

# --- config ---
MODEL_ID = "IPEC-COMMUNITY/spatialvla-4b-224-pt"
UNNORM_KEY = "bridge_orig/1.0.0"   # shipped stats/intrinsics; OK for a quick test
EP_DIR = "/home/homerobotics/beko/datasets/overfitbanana/episode_000/frames"
CACHE_DIR = None  # set to a path if you want a custom download/cache location

# --- load ---
processor = AutoProcessor.from_pretrained(
    MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR
)
model = AutoModel.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
).eval().cuda()

# --- iterate frames ---
frame_paths = sorted(glob.glob(os.path.join(EP_DIR, "*.npz")))

for i, p in enumerate(frame_paths):
    npz = np.load(p)
    # image: (H,W,3) uint8
    img = Image.fromarray(npz["wrist_image"])  # avoid HWC/CHW confusion
    # prompt: use per-frame task if present, else write your own
    task = npz["task"].item() if npz["task"].shape == () else str(npz["task"])
    prompt = task  # e.g., "pick the banana" or whatever your task string holds

    # build inputs; the processor inserts <image> tokens automatically if missing
    inputs = processor(images=[img], text=prompt, return_tensors="pt", unnorm_key=UNNORM_KEY)

    # run model → action tokens (Paligemma2-style generate; SpatialVLA adds predict_action())
    with torch.no_grad():
        gen = model.predict_action(inputs)  # returns only the newly generated tokens

    # decode → continuous actions (shape: [action_chunk_size, 7])
    out = processor.decode_actions(gen, unnorm_key=UNNORM_KEY)
    actions = out["actions"]          # numpy array, [4, 7] with this checkpoint
    action_ids = out["action_ids"]    # the 3*chunk token IDs

    # typical control: take the first action only
    a0 = actions[0]

        # --- debug prints ---
    if i > 50:
        print("DATASET DATA:")
        print("\n=== Frame:", os.path.basename(p), "===")
        print("Task:", task)
        print("State:", npz["state"])
        print("Actions (ground truth):", npz["actions"])

        print("PREDICTION:")
        print(os.path.basename(p), a0)

        # show the image in a blocking way
        plt.imshow(img)
        plt.title(f"{os.path.basename(p)} - {task}")
        plt.axis("off")
        plt.show()
