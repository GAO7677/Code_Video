"""CPU-only independent-history holdout diagnostic for Future Query Predictor."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT / "validation_20260920/paired_history_physics_v1"
POOL_MANIFEST_PATH = PROJECT / "validation_20260920/group_holdout_geometry_pools_v1/geometry_pool_manifest.json"
AUDIT_DIR = PROJECT / "validation_20260920/group_holdout_preflight_v1"
OUT = PROJECT / "validation_20260920/group_holdout_training_1500_v1"
TOKEN_COUNT = 1792
FEATURE_DIM = 1386
M = 512
FPS = 30.0
SEED = 20260920
TRAIN_SAMPLER_SEED = 20262020
EVAL_SEEDS = list(range(20262001, 20262017))
FIXED_EVAL_SEED = 20262000
STEPS = 1500
CHECKPOINT_STEPS = (0, 300, 750, 1500)
RADIUS = 0.18
PAIR_NONDEG_M = 0.05 * RADIUS
PAIR_R_THRESHOLD = 0.25
NEG_DELTA_THRESHOLD_M = 0.05 * RADIUS
STABILITY_THRESHOLD_M = 0.05 * RADIUS


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode()); h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def load_protocol():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    pool_manifest = json.loads(POOL_MANIFEST_PATH.read_text())
    data_manifest = json.loads((ROOT / "paired_history_data_manifest.json").read_text())
    records = sorted(data_manifest["records"], key=lambda r: (r["group_id"], r["width_m"]))
    pools = {r["key"]: r for r in pool_manifest["records"]}
    if len(records) != 24 or len(pools) != 24:
        raise ValueError("expected 24 episode records and 24 geometry pools")
    for r in records:
        if r["key"] not in pools or pools[r["key"]]["M"] != M:
            raise ValueError(f"missing or invalid pool {r['key']}")
    groups = {}
    for r in records:
        groups.setdefault(r["group_id"], []).append(r)
    for g in groups:
        groups[g].sort(key=lambda r: r["width_m"])
        if [r["width_m"] for r in groups[g]] != [0.46, 0.58, 0.7]:
            raise ValueError(f"group variants are not fixed for {g}")
    return data_manifest, pool_manifest, records, groups, pools


def load_episode(record):
    sample = ROOT / "samples" / record["key"]
    metadata = json.loads((sample / "metadata.json").read_text())
    with np.load(sample / "raw" / "states_xyzw.npz", allow_pickle=False) as data:
        states = {key: data[key].copy() for key in data.files}
    names = states["object_names"].astype(str).tolist()
    dynamic = [i for i, name in enumerate(names) if metadata["actors"][name].get("dynamic")]
    if dynamic != [0]:
        raise ValueError(f"expected dynamic object index 0: {record['key']}")
    actor = metadata["actors"][names[0]]
    if actor.get("shape") != "sphere":
        raise ValueError(f"expected sphere: {record['key']}")
    radius = float(actor["size_m"]["radius"])
    return {
        "record": record, "metadata": metadata, "states": states,
        "positions": states["positions"][:, 0].astype(np.float32),
        "observed": states["positions"][:8, 0].astype(np.float32),
        "target": states["positions"][8:49, 0].astype(np.float32),
        "times": states["frame_times"][:8].astype(np.float32),
        "radius_m": radius,
    }


def load_all(records):
    return {r["key"]: load_episode(r) for r in records}


def pool_for(pool_record):
    with np.load(pool_record["pool_path"], allow_pickle=False) as data:
        return (data["pool_xyz"].astype(np.float32), data["pool_collider"].astype(np.int16),
                data["pool_key_mask"].astype(bool), data["pool_names"].astype(str).tolist())


def sample_scene(pool_record, seed):
    sys.path.insert(0, str(PROJECT))
    from build_task_geometry_caches import choose
    xyz, collider, key_mask, names = pool_for(pool_record)
    ids = choose(xyz, collider, key_mask, names, int(seed), "dense_critical_task_geometry", M)
    if len(ids) != M or len(np.unique(ids)) != M:
        raise AssertionError("invalid dense task selection")
    scene_xyz = np.zeros((TOKEN_COUNT, 3), dtype=np.float32)
    scene_mask = np.zeros((TOKEN_COUNT,), dtype=bool)
    scene_xyz[:M] = xyz[ids]; scene_mask[:M] = True
    return scene_xyz, scene_mask, ids


def make_batch(rows, episodes, pools, sample_seeds):
    sys.path.insert(0, str(PROJECT))
    from future_query_predictor import motion_features
    positions = torch.from_numpy(np.stack([episodes[r["key"]]["observed"] for r in rows])[:, :, None])
    sizes = torch.from_numpy(np.asarray([[[episodes[r["key"]]["radius_m"] * 2] * 3] for r in rows], dtype=np.float32))
    times = torch.from_numpy(np.stack([episodes[r["key"]]["times"] for r in rows]))
    motion, last, velocity = motion_features(positions, sizes, times)
    b = len(rows)
    xyz = np.zeros((b, TOKEN_COUNT, 3), dtype=np.float32)
    mask = np.zeros((b, TOKEN_COUNT), dtype=bool)
    selected = []
    for i, (row, seed) in enumerate(zip(rows, sample_seeds)):
        sx, sm, ids = sample_scene(pools[row["key"]], int(seed))
        xyz[i] = sx; mask[i] = sm; selected.append(ids)
    batch = {
        "motion": motion.float(), "object_mask": torch.ones(b, 1, dtype=torch.bool),
        "last_position": last.float(), "last_velocity": velocity.float(),
        "future_dt": torch.arange(1, 42, dtype=torch.float32)[None].expand(b, -1) / FPS,
        "scene_xyz": torch.from_numpy(xyz),
        "scene_features": torch.zeros(b, TOKEN_COUNT, FEATURE_DIM, dtype=torch.float16),
        "scene_mask": torch.from_numpy(mask),
    }
    target = torch.from_numpy(np.stack([episodes[r["key"]]["target"] for r in rows])[:, None])
    if not all(torch.isfinite(value).all() for value in batch.values()) or not torch.isfinite(target).all():
        raise FloatingPointError("non-finite batch")
    return batch, target, selected


def subset(batch, target, i):
    idx = torch.tensor([i], dtype=torch.long)
    return {key: value[idx] for key, value in batch.items()}, target[idx]


def analytic_prediction(batch):
    return batch["last_position"][:, :, None] + batch["last_velocity"][:, :, None] * batch["future_dt"][:, None, :, None]


def group_seed(step, group_index, variant_index):
    # Fixed stream: both learned arms see precisely identical sampled points.
    return TRAIN_SAMPLER_SEED + step * 1009 + group_index * 17 + variant_index


def train_arm(name, scene_mode, train_groups, episodes, pools, mean, std, init_state, out_dir):
    sys.path.insert(0, str(PROJECT))
    from future_query_predictor import Predictor, objective
    model = Predictor(mean, std, scene_mode=scene_mode, seed=SEED, strict_future_grid=True).cpu()
    model.load_state_dict(init_state, strict=True)
    active = list(model.active_parameters())
    optimizer = torch.optim.AdamW(active, lr=3e-4, weight_decay=.01)
    checkpoints = out_dir / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    log = []
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(initial, checkpoints / f"{name}_step0000.pt")
    start = time.monotonic()
    sequence = []
    for i in range(250):
        sequence.extend(range(len(train_groups)))
    rng = np.random.default_rng(SEED)
    rng.shuffle(sequence)
    for step in range(1, STEPS + 1):
        gi = int(sequence[step - 1]); rows = train_groups[gi]
        sample_seeds = [group_seed(step, gi, vi) for vi in range(3)]
        batch, target, _ = make_batch(rows, episodes, pools, sample_seeds)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for vi in range(3):
            part, labels = subset(batch, target, vi)
            loss = objective(model(part), labels, part) / 3.0
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} nonfinite loss at step {step}")
            loss.backward(); losses.append(float(loss.detach()) * 3.0)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True))
        optimizer.step()
        total = float(sum(losses) / 3.0)
        log.append({"step": step, "group_id": rows[0]["group_id"], "widths_m": [r["width_m"] for r in rows],
                    "effective_batch": 3, "microbatches": 3, "sample_seeds": sample_seeds,
                    "loss": total, "gradient_norm_before_clip": grad_norm})
        if step in CHECKPOINT_STEPS[1:]:
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, checkpoints / f"{name}_step{step:04d}.pt")
            print("CHECKPOINT", name, step, total, flush=True)
    model.eval()
    with torch.no_grad():
        # A fixed training probe is useful for fit diagnosis and is not used
        # for checkpoint selection.
        probe_losses = []
        for gi, rows in enumerate(train_groups):
            batch, target, _ = make_batch(rows, episodes, pools, [FIXED_EVAL_SEED + gi * 11 + i for i in range(3)])
            probe_losses.append(float(objective(model(batch), target, batch)))
    result = {"name": name, "scene_mode": scene_mode, "steps": STEPS, "effective_batch": 3,
              "gradient_accumulation_microbatches": 3, "optimizer": "AdamW", "lr": 3e-4,
              "weight_decay": .01, "initial_state_sha256": state_sha(initial),
              "final_state_sha256": state_sha(model.state_dict()), "initial_loss": log[0]["loss"],
              "final_update_loss": log[-1]["loss"], "fixed_train_probe_loss": float(np.mean(probe_losses)),
              "elapsed_seconds": time.monotonic() - start,
              "active_parameters": int(sum(p.numel() for p in active)), "loss_log": log}
    return model, result, sequence


def boxes_and_radius(key, episodes):
    sys.path.insert(0, str(PROJECT))
    from penetration_decomposition import colliders_for
    colliders, radius = colliders_for(ROOT, key)
    boxes = [(c["center"], c["half"], c["quat"]) for c in colliders[1:]]
    return colliders, radius, boxes


def case_metrics(pred, target, episode, model_batch):
    sys.path.insert(0, str(PROJECT))
    from future_query_predictor import interval_velocity
    from penetration_decomposition import depth_stats, distances
    from trial_metrics import behaviour
    last = episode["observed"][-1]
    future_dt = np.arange(1, 42, dtype=np.float32) / FPS
    p = np.asarray(pred, dtype=np.float64); t = np.asarray(target, dtype=np.float64)
    error = np.linalg.norm(p - t, axis=-1)
    pv = interval_velocity(torch.from_numpy(p[None, None].astype(np.float32)),
                           torch.from_numpy(last[None, None].astype(np.float32)),
                           torch.from_numpy(future_dt[None])).numpy()[0, 0]
    tv = interval_velocity(torch.from_numpy(t[None, None].astype(np.float32)),
                           torch.from_numpy(last[None, None].astype(np.float32)),
                           torch.from_numpy(future_dt[None])).numpy()[0, 0]
    colliders, radius, boxes = boxes_and_radius(episode["record"]["key"], episodes=None)
    signed_pred = distances(p, radius, colliders); signed_gt = distances(t, radius, colliders)
    pred_depth = depth_stats(signed_pred.min(axis=1)); gt_depth = depth_stats(signed_gt.min(axis=1))
    event = behaviour(p.astype(np.float32), episode["observed"], t.astype(np.float32), radius, boxes)
    return {"ADE_m": float(error.mean()), "FDE_m": float(error[-1]), "radius_normalized_ADE": float(error.mean() / radius),
            "radius_normalized_FDE": float(error[-1] / radius), "interval_velocity_MAE_mps": float(np.abs(pv - tv).mean()),
            "position_loss": float(torch.nn.functional.smooth_l1_loss(torch.from_numpy(p.astype(np.float32)), torch.from_numpy(t.astype(np.float32)))),
            "velocity_loss": float(torch.nn.functional.smooth_l1_loss(torch.from_numpy(pv.astype(np.float32)), torch.from_numpy(tv.astype(np.float32)))),
            "physical_loss": float(torch.nn.functional.smooth_l1_loss(torch.from_numpy(p.astype(np.float32)), torch.from_numpy(t.astype(np.float32))) + .2 * torch.nn.functional.smooth_l1_loss(torch.from_numpy(pv.astype(np.float32)), torch.from_numpy(tv.astype(np.float32)))),
            "pred_max_penetration_m": pred_depth["max_depth_m"], "pred_primary_frame_rate": pred_depth["primary_frame_rate"],
            "pred_primary_longest_run": pred_depth["primary_longest_continuous_frames"], "gt_max_penetration_m": gt_depth["max_depth_m"],
            "pred_contact_onset_frame": event["pred_contact_onset_frame"], "gt_contact_onset_frame": event["gt_contact_onset_frame"],
            "contact_onset_classification_correct": event["contact_onset_classification_correct"],
            "contact_onset_time_error_s": event["contact_onset_time_error_s"],
            "post_contact_velocity_angle_deg": event["post_contact_velocity_angle_deg"]}


def pair_metrics(pa, pb, ta, tb, group, split, wa, wb, seed, arm):
    gt_delta = np.linalg.norm(ta - tb, axis=-1); pred_delta = np.linalg.norm(pa - pb, axis=-1)
    delta_error = np.linalg.norm((pa - pb) - (ta - tb), axis=-1)
    dgt, dpred, err = float(gt_delta.mean()), float(pred_delta.mean()), float(delta_error.mean())
    return {"arm": arm, "group_id": group, "split": split, "width_a_m": wa, "width_b_m": wb,
            "sampling_seed": seed, "D_gt_m": dgt, "D_pred_m": dpred, "E_delta_m": err,
            "R_delta": float(err / dgt) if dgt > 1e-8 else None,
            "nondegenerate": bool(dgt >= PAIR_NONDEG_M), "equal_future": bool(dgt < 1e-8),
            "response_pass": bool(dgt >= PAIR_NONDEG_M and err / dgt <= PAIR_R_THRESHOLD),
            "negative_pass": bool(dgt < 1e-8 and dpred <= NEG_DELTA_THRESHOLD_M)}


def evaluate(arms, records, groups, episodes, pools, out_dir):
    sys.path.insert(0, str(PROJECT))
    from future_query_predictor import Predictor
    ordered_groups = sorted(groups)
    group_indices = {g: i for i, g in enumerate(ordered_groups)}
    all_rows = records
    episode_index = {r["key"]: i for i, r in enumerate(all_rows)}
    preds = {name: np.zeros((len(EVAL_SEEDS), len(all_rows), 41, 3), dtype=np.float32) for name in arms}
    targets = np.stack([episodes[r["key"]]["target"] for r in all_rows]).astype(np.float32)
    fixed_preds = {name: np.zeros((len(all_rows), 41, 3), dtype=np.float32) for name in arms}
    case_rows, pair_rows, stability_rows = [], [], []
    seeds_all = [FIXED_EVAL_SEED] + EVAL_SEEDS
    for seed_role, seed_list in (("fixed_train_probe", [FIXED_EVAL_SEED]), ("new_sampling", EVAL_SEEDS)):
        for seed in seed_list:
            per_arm_preds = {name: np.zeros((len(all_rows), 41, 3), dtype=np.float32) for name in arms}
            for gi, group in enumerate(ordered_groups):
                rows = groups[group]
                sample_seeds = [int(seed + gi * 101 + vi) for vi in range(3)]
                batch, target, _ = make_batch(rows, episodes, pools, sample_seeds)
                for name, model in arms.items():
                    with torch.no_grad():
                        prediction = analytic_prediction(batch) if model is None else model(batch)
                    array = prediction.detach().cpu().numpy()[:, 0]
                    for vi, row in enumerate(rows):
                        per_arm_preds[name][episode_index[row["key"]]] = array[vi]
            for name, array in per_arm_preds.items():
                if seed_role == "new_sampling":
                    preds[name][EVAL_SEEDS.index(seed)] = array
                else:
                    fixed_preds[name] = array
                for i, row in enumerate(all_rows):
                    episode = episodes[row["key"]]
                    metric = case_metrics(array[i], episode["target"], episode, None)
                    case_rows.append({"arm": name, "seed_role": seed_role, "sampling_seed": seed,
                                      "key": row["key"], "group_id": row["group_id"], "split": row["split"],
                                      "width_m": row["width_m"], **metric})
            for group in ordered_groups:
                rows = groups[group]
                for left, right in itertools.combinations(range(3), 2):
                    a, b = rows[left], rows[right]
                    ia, ib = episode_index[a["key"]], episode_index[b["key"]]
                    for name, array in per_arm_preds.items():
                        pair_rows.append(pair_metrics(array[ia], array[ib], targets[ia], targets[ib], group, a["split"], a["width_m"], right and b["width_m"], seed, name) | {"seed_role": seed_role})
    # Sampling stability uses only the 16 predeclared new seeds, with no seed
    # selection.  It is summarized per episode, then by split/arm.
    for name, array in preds.items():
        for i, row in enumerate(all_rows):
            values = []
            for a, b in itertools.combinations(range(len(EVAL_SEEDS)), 2):
                values.append(float(np.linalg.norm(array[a, i] - array[b, i], axis=-1).mean()))
            stability_rows.append({"arm": name, "key": row["key"], "group_id": row["group_id"], "split": row["split"],
                                   "sampling_pair_count": len(values), "S_case_mean_m": float(np.mean(values)),
                                   "S_case_p95_m": float(np.quantile(values, .95)),
                                   "stability_pass": bool(np.mean(values) <= STABILITY_THRESHOLD_M)})
    np.savez_compressed(out_dir / "predictions.npz", targets=targets, eval_seeds=np.asarray(EVAL_SEEDS),
                        **{f"{name}_eval16": arr for name, arr in preds.items()},
                        **{f"{name}_fixed": arr for name, arr in fixed_preds.items()})
    return case_rows, pair_rows, stability_rows, preds, targets


def summarize(case_rows, pair_rows, stability_rows):
    summary = {"case": {}, "pair": {}, "stability": {}}
    for split in ("train", "dev"):
        for arm in sorted({r["arm"] for r in case_rows}):
            rows = [r for r in case_rows if r["split"] == split and r["arm"] == arm and r["seed_role"] == "new_sampling"]
            summary["case"][f"{split}/{arm}"] = {"rows": len(rows), "history_groups": len({r["group_id"] for r in rows}),
                "ADE_m_mean": float(np.mean([r["ADE_m"] for r in rows])), "FDE_m_mean": float(np.mean([r["FDE_m"] for r in rows])),
                "ADE_radius_mean": float(np.mean([r["radius_normalized_ADE"] for r in rows])),
                "interval_velocity_MAE_mps_mean": float(np.mean([r["interval_velocity_MAE_mps"] for r in rows])),
                "physical_loss_mean": float(np.mean([r["physical_loss"] for r in rows])),
                "contact_accuracy": float(np.mean([r["contact_onset_classification_correct"] for r in rows])),
                "contact_time_error_s_mean": float(np.mean([r["contact_onset_time_error_s"] for r in rows if r["contact_onset_time_error_s"] is not None])) if any(r["contact_onset_time_error_s"] is not None for r in rows) else None,
                "mean_max_penetration_mm": float(np.mean([r["pred_max_penetration_m"] for r in rows]) * 1000),
                "case_fraction_max_penetration_gt5mm": float(np.mean([r["pred_max_penetration_m"] > .005 for r in rows])),
                "mean_penetration_frame_rate": float(np.mean([r["pred_primary_frame_rate"] for r in rows]))}
    for split in ("train", "dev"):
        for arm in sorted({r["arm"] for r in pair_rows}):
            rows = [r for r in pair_rows if r["split"] == split and r["arm"] == arm and r["seed_role"] == "new_sampling"]
            pos = [r for r in rows if r["nondegenerate"]]
            neg = [r for r in rows if r["equal_future"]]
            summary["pair"][f"{split}/{arm}"] = {"rows": len(rows), "history_groups": len({r["group_id"] for r in rows}),
                "nondegenerate_rows": len(pos), "nondegenerate_history_groups": len({r["group_id"] for r in pos}),
                "response_pass_fraction": float(np.mean([r["response_pass"] for r in pos])) if pos else None,
                "mean_R_delta": float(np.mean([r["R_delta"] for r in pos])) if pos else None,
                "equal_future_rows": len(neg), "negative_pass_fraction": float(np.mean([r["negative_pass"] for r in neg])) if neg else None,
                "negative_D_pred_mean_m": float(np.mean([r["D_pred_m"] for r in neg])) if neg else None}
    for split in ("train", "dev"):
        for arm in sorted({r["arm"] for r in stability_rows}):
            rows = [r for r in stability_rows if r["split"] == split and r["arm"] == arm]
            summary["stability"][f"{split}/{arm}"] = {"episodes": len(rows), "history_groups": len({r["group_id"] for r in rows}),
                "S_case_mean_m": float(np.mean([r["S_case_mean_m"] for r in rows])),
                "S_case_p95_m": float(np.quantile([r["S_case_mean_m"] for r in rows], .95)),
                "pass_fraction": float(np.mean([r["stability_pass"] for r in rows]))}
    return summary


def make_report(summary, train_results, protocol, out_dir):
    dev_pos = {k: v for k, v in summary["pair"].items() if k.startswith("dev/") and v["nondegenerate_rows"]}
    report = {
        "schema": "independent_history_group_holdout_report_v1", "status": "EXECUTED", "formal_training": False,
        "device": "cpu", "threads": 2, "steps": STEPS, "effective_batch": 3,
        "arms": ["analytic_cv", "motion_only", "geometry_dense_resample"], "train_results": train_results,
        "summary": summary, "protocol": protocol,
        "main_answer": {
            "new_history_scene_branch_better_than_motion_only": "NOT_DETERMINABLE_POSITIVE_SCENE_RESPONSE: g06/g07 have zero nondegenerate future pairs; compare ordinary errors only with this limitation",
            "equal_future_control": "reported; geometry must remain invariant on g06/g07 equal-future pairs",
            "failure_localization": "requires separating train fit from dev transfer; contact/penetration is reported independently and is not implied by ADE",
            "next_priority": "generate at least one independent dev history per group with nondegenerate door-width response, then repeat frozen sampling protocol; do not modify network based on this degenerate dev split",
        },
        "interpretation_limits": [
            "Task geometry is privileged declared collision geometry, not RGB-visible and not a formal visual/OOD result.",
            "Six train groups are independent histories; 16 sampling seeds are repeated evaluations, not independent histories.",
            "No checkpoint was selected by dev and no extra steps were run.",
            "The original Bullet door replay matched all 24 saved episodes exactly and reported zero rebound corrections.",
        ],
    }
    write_json(out_dir / "report.json", report)
    def fmt(x):
        if x is None: return "N/A"
        if isinstance(x, float): return f"{x:.5f}"
        return str(x)
    lines = ["# Independent History Group Holdout", "", "Status: `EXECUTED` / CPU diagnostic only", "", "## Main result", "", 
             "g06/g07 are the predeclared development histories, but every width pair in both groups has D_gt=0. They therefore test equal-future invariance only; they cannot support a positive scene-response claim.", "", "## Metrics (16 new sampling seeds)", "", "| split/arm | ADE (m) | FDE (m) | ADE/r | interval-v MAE | physical loss | >5mm case fraction | mean max penetration (mm) | contact accuracy |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, row in sorted(summary["case"].items()):
        lines.append("| " + key + " | " + " | ".join(fmt(row.get(k)) for k in ("ADE_m_mean", "FDE_m_mean", "ADE_radius_mean", "interval_velocity_MAE_mps_mean", "physical_loss_mean", "case_fraction_max_penetration_gt5mm", "mean_max_penetration_mm", "contact_accuracy")) + " |")
    lines += ["", "## Pair response", "", "| split/arm | nondegenerate rows | nondegenerate groups | response pass | mean R | equal-future rows | negative pass | mean negative D_pred (m) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, row in sorted(summary["pair"].items()):
        lines.append("| " + key + " | " + " | ".join(fmt(row.get(k)) for k in ("nondegenerate_rows", "nondegenerate_history_groups", "response_pass_fraction", "mean_R_delta", "equal_future_rows", "negative_pass_fraction", "negative_D_pred_mean_m")) + " |")
    lines += ["", "## Sampling stability", "", "| split/arm | episodes | mean S_case (m) | p95 S_case (m) | pass fraction |", "|---|---:|---:|---:|---:|"]
    for key, row in sorted(summary["stability"].items()):
        lines.append("| " + key + " | " + " | ".join(fmt(row.get(k)) for k in ("episodes", "S_case_mean_m", "S_case_p95_m", "pass_fraction")) + " |")
    lines += ["", "## Interpretation", "", "- `geometry_dense_resample` and `motion_only` use the same new initialization, group order, loss, 1500-step budget, and per-step three-width effective batch.", "- Pair and stability seed counts are repeated measurements; they are not additional histories.", "- The task geometry is privileged and excludes Utonia/RGB; no visual generalization or layout-OOD claim is made.", "- The next experiment should add nondegenerate independent development histories before changing the predictor or declaring scene conditioning successful."]
    (out_dir / "group_holdout_report.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    torch.set_num_threads(2)
    data_manifest, pool_manifest, records, groups, pools = load_protocol()
    OUT.mkdir(parents=True)
    episodes = load_all(records)
    stats_path = AUDIT_DIR / "train_motion_stats.npz"
    with np.load(stats_path, allow_pickle=False) as stats:
        mean = torch.from_numpy(stats["mean"].astype(np.float32)); std = torch.from_numpy(stats["std"].astype(np.float32))
    train_groups = [groups[f"history_door_g{i:02d}"] for i in range(6)]
    sys.path.insert(0, str(PROJECT))
    from future_query_predictor import Predictor
    base = Predictor(mean, std, scene_mode="geometry_only", seed=SEED, strict_future_grid=True).cpu()
    init_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    train_order = np.repeat(np.arange(6), 250); np.random.default_rng(SEED).shuffle(train_order)
    protocol = {
        "schema": "independent_history_group_holdout_protocol_v1", "status": "EXECUTED",
        "data_manifest": str(ROOT / "paired_history_data_manifest.json"), "pool_manifest": str(POOL_MANIFEST_PATH),
        "train_groups": [f"history_door_g{i:02d}" for i in range(6)], "dev_groups": ["history_door_g06", "history_door_g07"],
        "widths_m": [0.46, 0.58, 0.7], "episodes": 24, "M": M, "padded_tokens": TOKEN_COUNT,
        "scene_mode": "dense_critical_task_geometry", "scene_features": "zero placeholder ignored by geometry_only",
        "future_rgb_or_state_used_for_scene": False, "dit_loaded": False, "utonia_extracted": False,
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": .01}, "steps": STEPS,
        "effective_batch": 3, "microbatch": 1, "gradient_accumulation_microbatches": 3,
        "loss": "SmoothL1(position)+0.2*SmoothL1(interval_velocity)", "query_reference": "legacy_last_position",
        "seed": SEED, "train_sampler_seed": TRAIN_SAMPLER_SEED, "eval_seeds": EVAL_SEEDS,
        "fixed_eval_seed": FIXED_EVAL_SEED, "train_group_sequence_sha256": hashlib.sha256(train_order.tobytes()).hexdigest(),
        "initial_state_sha256": state_sha(init_state), "train_motion_stats_sha256": sha(stats_path),
        "thresholds": {"pair_non_degenerate_m": PAIR_NONDEG_M, "pair_R": PAIR_R_THRESHOLD,
                       "negative_D_pred_m": NEG_DELTA_THRESHOLD_M, "stability_m": STABILITY_THRESHOLD_M, "penetration_primary_m": .005},
        "no_dev_checkpoint_selection": True, "no_auto_extension": True,
    }
    write_json(OUT / "protocol.json", protocol)
    arms = {}
    train_results = {}
    for name, mode in (("motion_only", "motion_only"), ("geometry_dense_resample", "geometry_only")):
        model, result, sequence = train_arm(name, mode, train_groups, episodes, pools, mean, std, init_state, OUT)
        arms[name] = model; train_results[name] = result
        write_json(OUT / f"{name}_train_result.json", result)
    arms["analytic_cv"] = None
    case_rows, pair_rows, stability_rows, preds, targets = evaluate(arms, records, groups, episodes, pools, OUT)
    write_csv(OUT / "per_case_eval.csv", case_rows); write_csv(OUT / "pair_eval.csv", pair_rows); write_csv(OUT / "sampling_stability.csv", stability_rows)
    summary = summarize(case_rows, pair_rows, stability_rows)
    report = make_report(summary, train_results, protocol, OUT)
    print(json.dumps({"status": report["status"], "output": str(OUT), "dev_pair_non_degenerate": {k:v["nondegenerate_rows"] for k,v in summary["pair"].items() if k.startswith("dev/")}, "case_summary": summary["case"]}, indent=2))


if __name__ == "__main__":
    main()
