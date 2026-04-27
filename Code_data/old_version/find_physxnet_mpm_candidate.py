from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SOFT_KEYWORDS = {
    "foam": ("mpm_elastic", 3.0),
    "fabric": ("pbd_or_mpm", 2.5),
    "cloth": ("pbd_or_mpm", 2.5),
    "leather": ("mpm_elastic", 1.8),
    "rubber": ("mpm_elastic", 3.0),
    "silicone": ("mpm_elastic", 3.0),
    "gel": ("mpm_elastic", 3.2),
    "sponge": ("mpm_elastic", 3.0),
    "plasticine": ("mpm_elastoplastic", 3.4),
    "clay": ("mpm_elastoplastic", 3.4),
    "putty": ("mpm_elastoplastic", 3.4),
    "wax": ("mpm_elastoplastic", 2.8),
    "latex": ("mpm_elastic", 2.8),
}

RIGID_KEYWORDS = {
    "wood", "metal", "glass", "ceramic", "stone", "concrete", "steel", "iron", "aluminum"
}


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_density_to_kgm3(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().lower().replace(",", "")
    direct = safe_float(s)
    if direct is not None:
        return direct
    if "g/cm" in s:
        num = safe_float(s.split()[0])
        return None if num is None else num * 1000.0
    if "kg/m" in s:
        num = safe_float(s.split()[0])
        return None if num is None else num
    return None


def parse_youngs_modulus_pa(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().lower().replace(",", "")
    num = safe_float(s.split()[0])
    if num is None:
        return None
    if "gpa" in s or "young's modulus (gpa)" in s:
        return num * 1e9
    if "mpa" in s:
        return num * 1e6
    if "kpa" in s:
        return num * 1e3
    # PhysXNet examples often store raw number under a GPa field.
    return num * 1e9


@dataclass
class PartCandidate:
    label: int
    name: str
    material: str
    density_kgm3: Optional[float]
    youngs_pa: Optional[float]
    poisson: Optional[float]
    solver: str
    score: float


@dataclass
class ObjectCandidate:
    object_id: str
    object_name: str
    category: str
    dimension: str
    score: float
    soft_parts: List[PartCandidate]
    all_materials: List[str]


def classify_part(part: Dict[str, Any]) -> PartCandidate:
    material = str(part.get("material", "")).strip()
    material_low = material.lower()
    score = 0.0
    solver = "rigid"

    for kw, (family, bonus) in SOFT_KEYWORDS.items():
        if kw in material_low:
            solver = family
            score += bonus

    if solver == "rigid":
        e_pa = parse_youngs_modulus_pa(part.get("Young's Modulus (GPa)"))
        if e_pa is not None:
            # Soft-ish if not extremely stiff.
            if e_pa <= 2e8:
                solver = "mpm_elastic"
                score += 2.2
            elif e_pa <= 1e9 and not any(k in material_low for k in RIGID_KEYWORDS):
                solver = "mpm_elastic"
                score += 1.0

    density = parse_density_to_kgm3(part.get("density"))
    youngs_pa = parse_youngs_modulus_pa(part.get("Young's Modulus (GPa)"))
    poisson = safe_float(part.get("Poisson's Ratio"))

    if density is not None:
        if density <= 500:
            score += 0.8
        elif density <= 900:
            score += 0.3

    move_desc = str(part.get("Movement_description", "")).lower()
    if any(k in move_desc for k in ["deform", "bend", "compress", "stretch", "squeeze"]):
        score += 1.0
        if solver == "rigid":
            solver = "mpm_elastic"

    return PartCandidate(
        label=int(part.get("label", -1)),
        name=str(part.get("name", "")),
        material=material,
        density_kgm3=density,
        youngs_pa=youngs_pa,
        poisson=poisson,
        solver=solver,
        score=score,
    )


def scan_dataset(physx_root: Path, version: str) -> List[ObjectCandidate]:
    finaljson_dir = physx_root / version / "finaljson"
    if not finaljson_dir.exists():
        raise FileNotFoundError(f"finaljson dir not found: {finaljson_dir}")

    results: List[ObjectCandidate] = []
    for jp in sorted(finaljson_dir.glob("*.json")):
        try:
            meta = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue

        part_candidates = [classify_part(p) for p in meta.get("parts", [])]
        soft_parts = [p for p in part_candidates if p.solver != "rigid" and p.score > 0.0]
        if not soft_parts:
            continue

        obj_score = sum(p.score for p in soft_parts)
        if any(p.solver == "mpm_elastoplastic" for p in soft_parts):
            obj_score += 0.5
        if any(p.solver == "mpm_elastic" for p in soft_parts):
            obj_score += 0.5

        results.append(
            ObjectCandidate(
                object_id=jp.stem,
                object_name=str(meta.get("object_name", "")),
                category=str(meta.get("category", "")),
                dimension=str(meta.get("dimension", "")),
                score=obj_score,
                soft_parts=sorted(soft_parts, key=lambda x: x.score, reverse=True),
                all_materials=sorted({str(p.get("material", "")).strip() for p in meta.get("parts", []) if str(p.get("material", "")).strip()}),
            )
        )

    results.sort(key=lambda x: x.score, reverse=True)
    return results


def write_report(cands: List[ObjectCandidate], out_dir: Path, topk: int = 50) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "mpm_candidates_top.json"
    csv_path = out_dir / "mpm_candidates_top.csv"

    json_path.write_text(
        json.dumps([
            {
                **{k: v for k, v in asdict(c).items() if k != "soft_parts"},
                "soft_parts": [asdict(p) for p in c.soft_parts],
            }
            for c in cands[:topk]
        ], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "object_id", "object_name", "category", "score", "dimension", "materials", "soft_parts"])
        for i, c in enumerate(cands[:topk], 1):
            soft_desc = "; ".join(
                f"label={p.label}|name={p.name}|mat={p.material}|solver={p.solver}|score={p.score:.2f}|rho={p.density_kgm3}|E={p.youngs_pa}"
                for p in c.soft_parts
            )
            w.writerow([i, c.object_id, c.object_name, c.category, f"{c.score:.3f}", c.dimension, ", ".join(c.all_materials), soft_desc])
    return json_path, csv_path


def print_best(best: ObjectCandidate, out_dir: Path) -> None:
    print("=" * 80)
    print("推荐 MPM 候选物体")
    print("=" * 80)
    print(f"object_id   : {best.object_id}")
    print(f"object_name : {best.object_name}")
    print(f"category    : {best.category}")
    print(f"dimension   : {best.dimension}")
    print(f"score       : {best.score:.3f}")
    print(f"materials   : {', '.join(best.all_materials)}")
    print("soft parts   :")
    for p in best.soft_parts:
        print(
            f"  - label={p.label} name={p.name} material={p.material} solver={p.solver} "
            f"score={p.score:.2f} rho={p.density_kgm3} E={p.youngs_pa} nu={p.poisson}"
        )
    print("\n建议：")
    print("1) 若 solver=mpm_elastic，可先按 Neo-Hookean/Corotated 弹性体测试。")
    print("2) 若 solver=mpm_elastoplastic，可先测塑性/屈服版本。")
    print("3) 若同时混有 rigid part，可把 rigid part 保持 URDF，soft part 单独做 voxel/particle fill 后进 Genesis MPM。")
    print("\n已保存：")
    print(f"- {out_dir / 'mpm_candidates_top.json'}")
    print(f"- {out_dir / 'mpm_candidates_top.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="遍历 PhysXNet，挑出最适合做 MPM 预览的对象")
    parser.add_argument("--physx_root", type=Path, required=True)
    parser.add_argument("--version", type=str, default="version_1")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    cands = scan_dataset(args.physx_root, args.version)
    if not cands:
        raise RuntimeError("没有找到明显适合 MPM 的候选。可以把关键词表扩展到你自己的材料命名。")

    out_dir = args.output_root
    write_report(cands, out_dir, topk=args.topk)
    print_best(cands[0], out_dir)


if __name__ == "__main__":
    main()
'''

python /home/gaoya/Code_Video/Code_data/find_physxnet_mpm_candidate.py \
  --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
  --version version_1 \
  --output_root /data/gaoya/AAA_test_video/Dataset_test/physxnet_mpm_scan \
  --topk 20


'''