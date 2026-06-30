"""
构建物理视频比较 manifest，从三个数据集各抽50条：
1. phyco_kubric (rgba.mp4 from tar)
2. pybullet val (source_video.mp4)
3. physics-iq benchmark (30FPS full videos)

输出: /data/gaoya/agent-data/outputs/phys_compare_manifest.json
每条: {video_path, caption, source, scenario}
"""
import os
import json
import random
import tarfile
import re
import shutil
from pathlib import Path

random.seed(42)

OUT_DIR = "/data/gaoya/agent-data/outputs/phys_compare"
MANIFEST_PATH = "/data/gaoya/agent-data/outputs/phys_compare_manifest.json"
PHYCO_EXTRACT_DIR = os.path.join(OUT_DIR, "phyco_kubric_rgba")

os.makedirs(PHYCO_EXTRACT_DIR, exist_ok=True)

records = []

# ============================================================
# 1. phyco_kubric: 8 scenarios with caption.txt, rgba in tar
# ============================================================
PHYCO_BASE = "/data/gaoya/dataset/nnsriram97-phyco_kubric"
PHYCO_SCENARIOS = [
    "ball_drop_soft_v4",
    "ball_drop_v2",
    "ball_wall_collision",
    "cube_deform_soft_v2_noeff",
    "friction_slide_flat_force_v3",
    "friction_slide_flat_v2",
    "jenga_force",
    "pool_table_force",
]
TARGET_PER_SCENARIO = 7  # 8 scenarios * 7 ≈ 56, will trim to 50

print("=== phyco_kubric ===")
for scenario in PHYCO_SCENARIOS:
    sdir = os.path.join(PHYCO_BASE, scenario)
    caption_file = os.path.join(sdir, "common_caption_cosmos.txt")
    if not os.path.isfile(caption_file):
        print(f"  SKIP {scenario}: no caption.txt")
        continue

    caption = open(caption_file).read().strip()

    # find tar files
    tar_files = sorted([f for f in os.listdir(sdir) if f.endswith(".tar.gz")])
    if not tar_files:
        print(f"  SKIP {scenario}: no tar.gz")
        continue

    # scan first tar, iterate up to SCAN_LIMIT members, then sample
    SCAN_LIMIT = 300
    tpath = os.path.join(sdir, tar_files[0])
    pool = []  # list of tarfile.TarInfo
    try:
        with tarfile.open(tpath, "r:gz") as tar:
            for m in tar:
                if m.name.endswith("rgba.mp4"):
                    pool.append(m)
                    if len(pool) >= SCAN_LIMIT:
                        break

            if not pool:
                print(f"  SKIP {scenario}: no rgba in first tar")
                continue

            sampled_members = random.sample(pool, min(TARGET_PER_SCENARIO, len(pool)))
            print(f"  {scenario}: scanned {len(pool)} in pool, sampling {len(sampled_members)}")

            for m in sampled_members:
                parts = m.name.split("/")
                hash_id = parts[1] if len(parts) >= 2 else "unknown"
                out_path = os.path.join(PHYCO_EXTRACT_DIR, scenario, hash_id, "rgba.mp4")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                if not os.path.isfile(out_path):
                    f = tar.extractfile(m)
                    if f is None:
                        continue
                    with open(out_path, "wb") as wf:
                        wf.write(f.read())

                records.append({
                    "video_path": out_path,
                    "caption": caption,
                    "source": "phyco_kubric",
                    "scenario": scenario,
                    "original_tar": tpath,
                    "hash_id": hash_id,
                })
    except Exception as e:
        print(f"  WARN: failed {tpath}: {e}")

# trim to 50
phyco_records = records[:50]
records = phyco_records
print(f"phyco_kubric total: {len(records)}")

# ============================================================
# 2. pybullet val: source_video.mp4 + input_prompt
# ============================================================
PYBULLET_BASE = "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val"
PYBULLET_FAMILIES = ["F1_single_object", "F2_two_object", "F3_chain_reaction", "F4_occlusion", "F5_drop_support"]

print("\n=== pybullet ===")
pybullet_all = []
for family in PYBULLET_FAMILIES:
    fdir = os.path.join(PYBULLET_BASE, family)
    if not os.path.isdir(fdir):
        continue
    samples = sorted(os.listdir(fdir))
    for s in samples:
        meta_path = os.path.join(fdir, s, "meta.json")
        video_path = os.path.join(fdir, s, "source_video.mp4")
        if not os.path.isfile(meta_path) or not os.path.isfile(video_path):
            continue
        try:
            meta = json.load(open(meta_path))
            caption = meta.get("input_prompt") or meta.get("description") or meta.get("caption", "")
            if not caption:
                continue
            pybullet_all.append({
                "video_path": video_path,
                "caption": caption,
                "source": "pybullet",
                "scenario": meta.get("family_slug") or family,
                "sample_id": s,
            })
        except Exception as e:
            print(f"  WARN {s}: {e}")

sampled_pybullet = random.sample(pybullet_all, min(50, len(pybullet_all)))
print(f"pybullet: {len(pybullet_all)} total, sampling {len(sampled_pybullet)}")
records.extend(sampled_pybullet)

# ============================================================
# 3. physics-iq: mytest meta.json + 30FPS full videos
# ============================================================
PHYSIQ_MYTEST = "/data/gaoya/dataset/physics-iq-benchmark/mytest"
PHYSIQ_30FPS = "/data/gaoya/dataset/physics-iq-benchmark/full-videos/take-1/30FPS"

print("\n=== physics-iq ===")
physiq_all = []
if os.path.isdir(PHYSIQ_MYTEST):
    for case_dir in sorted(os.listdir(PHYSIQ_MYTEST)):
        meta_path = os.path.join(PHYSIQ_MYTEST, case_dir, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            meta = json.load(open(meta_path))
            caption = meta.get("caption") or meta.get("description") or ""
            if not caption:
                continue
            # case_dir: e.g. 0020_perspective-center_trimmed-ball-ramp
            # map to 30FPS: 0020_full-videos_30FPS_perspective-center_take-1_trimmed-ball-ramp.mp4
            m = re.match(r"(\d+)_(.+?)_trimmed-(.+)", case_dir)
            if not m:
                continue
            idx, perspective, scene = m.group(1), m.group(2), m.group(3)
            # find matching 30FPS video
            target = f"{idx}_full-videos_30FPS_{perspective}_take-1_trimmed-{scene}.mp4"
            video_path = os.path.join(PHYSIQ_30FPS, target)
            if not os.path.isfile(video_path):
                # try any matching file
                matches = [f for f in os.listdir(PHYSIQ_30FPS) if f.startswith(idx + "_")]
                if matches:
                    video_path = os.path.join(PHYSIQ_30FPS, matches[0])
                else:
                    print(f"  WARN: no 30FPS video for {case_dir}")
                    continue
            physiq_all.append({
                "video_path": video_path,
                "caption": caption,
                "source": "physics-iq",
                "scenario": meta.get("scenario") or meta.get("category") or scene,
                "sample_id": case_dir,
                "perspective": perspective,
            })
        except Exception as e:
            print(f"  WARN {case_dir}: {e}")

sampled_physiq = random.sample(physiq_all, min(50, len(physiq_all)))
print(f"physics-iq: {len(physiq_all)} total, sampling {len(sampled_physiq)}")
records.extend(sampled_physiq)

# ============================================================
# Write manifest
# ============================================================
print(f"\nTotal records: {len(records)}")
print(f"  phyco_kubric: {sum(1 for r in records if r['source']=='phyco_kubric')}")
print(f"  pybullet:     {sum(1 for r in records if r['source']=='pybullet')}")
print(f"  physics-iq:   {sum(1 for r in records if r['source']=='physics-iq')}")

# verify all video files exist
missing = [r for r in records if not os.path.isfile(r["video_path"])]
if missing:
    print(f"\nWARN: {len(missing)} missing video files!")
    for r in missing[:5]:
        print(f"  {r['video_path']}")

with open(MANIFEST_PATH, "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
print(f"\nManifest saved to: {MANIFEST_PATH}")
