#!/usr/bin/env python3
"""Evaluate main and experiment-specific losses on the fixed 30-case manifest."""
from __future__ import annotations
import argparse, copy, json, os, statistics, time
from pathlib import Path
import torch

import evaluate_train_subset_val_loss as ev

def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def normalize_case_seeds(manifest: dict) -> tuple[dict, bool]:
    normalized = copy.deepcopy(manifest)
    global_seed = int(normalized["seed"])
    changed = False
    for case in normalized["cases"]:
        if "case_seed" in case:
            continue
        case["case_seed"] = ev.stable_case_seed(
            global_seed,
            str(case["source"]),
            int(case["source_index"]),
        )
        changed = True
    return normalized, changed

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--gpu",type=int,required=True); p.add_argument("--entry-id"); args=p.parse_args()
    if args.gpu == 4: raise SystemExit("GPU4 prohibited")
    cfg=json.loads(args.config.read_text()); root=Path(cfg["output_root"]); case_manifest_path=Path(cfg["cases_manifest"]); case_manifest=json.loads(case_manifest_path.read_text())
    case_manifest, manifest_changed = normalize_case_seeds(case_manifest)
    if manifest_changed:
        atomic(case_manifest_path, case_manifest)
    entries=[e for e in cfg["entries"] if not args.entry_id or e["entry_id"]==args.entry_id]
    status_path=root/"loss_status.json"; status=json.loads(status_path.read_text()) if status_path.is_file() else {"state":"running","entries":{}}
    total_cases = len(case_manifest["cases"])
    for configured_entry in cfg["entries"]:
        entry_id = configured_entry["entry_id"]
        status.setdefault("entries", {}).setdefault(entry_id, {
            "state": "pending",
            "completed_cases": 0,
            "total_cases": total_cases,
            "checkpoint": configured_entry.get("checkpoint"),
        })
    status["state"] = "complete" if all(
        value.get("state") == "complete" for value in status["entries"].values()
    ) else "partial"
    atomic(status_path, status)
    datasets=None
    for entry in entries:
        out=root/"losses"/f"{entry['entry_id']}.json"
        if out.is_file():
            old=json.loads(out.read_text()); done=len(old.get("cases",[]))
            if old.get("state")=="complete" and done==len(case_manifest["cases"]): status["entries"][entry["entry_id"]]={"state":"complete","completed_cases":done,"total_cases":done}; atomic(status_path,status); continue
        checkpoint = entry.get("checkpoint")
        checkpoint_path = Path(str(checkpoint)) if checkpoint else None
        if checkpoint_path is None or not checkpoint_path.exists():
            error = f"checkpoint not found: {checkpoint or '<unset>'}"
            status["entries"][entry["entry_id"]] = {
                "state": "blocked",
                "completed_cases": 0,
                "total_cases": total_cases,
                "checkpoint": checkpoint,
                "error": error,
            }
            atomic(status_path, status)
            print(f"[{entry['entry_id']}] BLOCKED: {error}", flush=True)
            continue
        status["entries"][entry["entry_id"]]={"state":"running","completed_cases":0,"total_cases":len(case_manifest["cases"]),"checkpoint":entry["checkpoint"]}; status["current_entry"]=entry["entry_id"]; atomic(status_path,status)
        manifest_path=Path(entry["config"])
        kind=ev.model_kind(ev.load_resolved(manifest_path))
        device=torch.device("cuda:0")
        model,args_ns,config,kind=ev.build_model(manifest_path,device)
        # Ensure cached PyBullet tensors are used whenever available.
        cache_root=str(config["paths"].get("pybullet_root", "/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5"))
        for name, suffix in (("pybullet0713_vae_cache_dir","vae_latents_wan22_512x896_49f_prefix_bf16"),("pybullet0713_prompt_cache_dir","prompt_embeddings_wan22_umt5_bf16")):
            value=getattr(args_ns,name,None)
            if value is None or not str(value): setattr(args_ns,name,str(Path(cache_root)/suffix))
        datasets=ev.build_source_datasets(config)
        trajectory_cache_store=ev.build_trajectory_cache(config)
        load_info=ev.load_checkpoint(model,Path(entry["checkpoint"]))
        result={"schema_version":1,"entry_id":entry["entry_id"],"method_label":entry["method_label"],"version":entry["version"],"step":entry["step"],"checkpoint":entry["checkpoint"],"config":entry["config"],"metric":"fixed_pybullet_train_30case_eval","state":"running","cases":[]}
        for case in case_manifest["cases"]:
            prepared=ev.prepare_inputs(model,datasets[case["source"]],case,root)
            trajectory_cache=(
                trajectory_cache_store.load(str(case["sample_key"]))
                if trajectory_cache_store is not None
                else None
            )
            loss,metrics=ev.evaluate_prepared(
                model,
                prepared,
                int(case.get("case_seed",42)),
                trajectory_cache=trajectory_cache,
            )
            result["cases"].append({**case,"loss_main":loss,"metrics":metrics})
            atomic(out,result)
            print(f"[{entry['entry_id']}] {len(result['cases'])}/{len(case_manifest['cases'])} main={loss:.8f}",flush=True)
        result["load_info"]=load_info; result["state"]="complete"; result["completed_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); atomic(out,result)
        del model; torch.cuda.empty_cache()
        status["entries"][entry["entry_id"]]={"state":"complete","completed_cases":len(result["cases"]),"total_cases":len(result["cases"])}; atomic(status_path,status)
    status.pop("current_entry",None); status["state"]="complete" if all(v.get("state")=="complete" for v in status.get("entries",{}).values()) else "partial"; atomic(status_path,status)

if __name__=="__main__": main()
