#!/usr/bin/env python3
"""Run fixed 30-case video inference for six explicitly listed checkpoints."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path

PYTHON = "/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "run_infer_from_experiment.sh"

def completed_cases(out: Path, cases: list[dict]) -> int:
    total = 0
    for case in cases:
        matches = list(out.glob(f"{case['case_id']}*.mp4")) + list(out.glob(f"*/{case['case_id']}*.mp4"))
        if any(path.is_file() and path.stat().st_size > 0 for path in matches):
            total += 1
    return total

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--gpu",type=int,required=True); p.add_argument("--entry-id"); args=p.parse_args()
    if args.gpu == 4: raise SystemExit("GPU4 prohibited")
    cfg=json.loads(args.config.read_text()); root=Path(cfg["output_root"]); manifest=json.loads(Path(cfg["cases_manifest"]).read_text())
    input_list=root / "inputs" / "cases.txt"; log_root=root / "video_logs"; log_root.mkdir(parents=True,exist_ok=True)
    entries=[e for e in cfg["entries"] if not args.entry_id or e["entry_id"]==args.entry_id]
    status_path=root/"video_status.json"; status=json.loads(status_path.read_text()) if status_path.is_file() else {"state":"running","entries":{}}
    for e in entries:
        out=root/"videos"/e["entry_id"]; done=completed_cases(out, manifest["cases"])
        if done==len(manifest["cases"]): status["entries"][e["entry_id"]]={"state":"complete","completed_cases":done,"total_cases":len(manifest["cases"])}; continue
        status["entries"][e["entry_id"]]={"state":"running","completed_cases":done,"total_cases":len(manifest["cases"]),"checkpoint":e["checkpoint"]}; status["current_entry"]=e["entry_id"]; status["updated_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
        env=os.environ.copy(); env.update(PYTHONNOUSERSITE="1",TEST_LIST=str(input_list),NUM_INFERENCE_STEPS=str(cfg["inference"]["num_inference_steps"]),EXPERIMENT_CONFIG=e["config"],STEP_OUTPUT_DIR_NAME=e["entry_id"],SHARD_TAG=e["entry_id"],FORCE_INFERENCE="1")
        log=(log_root/f"{e['entry_id']}.log").open("a",encoding="utf-8")
        rc=subprocess.run(["bash",str(RUN),e["checkpoint"],str(args.gpu),str(out)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode; log.close()
        done=completed_cases(out, manifest["cases"])
        status["entries"][e["entry_id"]]={"state":"complete" if rc==0 and done==len(manifest["cases"]) else "failed","completed_cases":done,"total_cases":len(manifest["cases"]),"return_code":rc}; status["updated_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
    status.pop("current_entry",None); status["state"]="complete" if all(v.get("state")=="complete" for v in status["entries"].values()) else "partial"; status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")

if __name__=="__main__": main()
