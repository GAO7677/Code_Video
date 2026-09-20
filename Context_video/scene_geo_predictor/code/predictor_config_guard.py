"""Explicit diagnostic/formal dependency checks for new predictor experiments."""
from __future__ import annotations

import json
from pathlib import Path

IMPLEMENTED_MODES = {"analytic_cv", "motion_only", "constant_legacy", "geometry_only",
                     "geometry_utonia", "oracle_visible_geometry", "oracle_partial_visible_geometry"}
UNIMPLEMENTED_SWITCHES = {"two_pass_future_query", "relative_motion_features", "unit_normalized_loss"}


def _read(path):
    path = Path(path)
    return json.loads(path.read_text())


def validate(config_path, *, protocol: str, variant: str, scene_reports=(), allow_legacy=False):
    if protocol not in {"diagnostic", "formal"}:
        raise ValueError("protocol must be diagnostic or formal")
    config = _read(config_path) if isinstance(config_path, (str, Path)) else config_path
    if config.get("schema") != "future_query_predictor_experiment_v1":
        raise ValueError("unsupported predictor configuration schema")
    switches = config.get("experimental_switches_default", {})
    unknown = set(switches) - {"reference_mode", *UNIMPLEMENTED_SWITCHES}
    if unknown:
        raise ValueError(f"unknown experimental switches: {sorted(unknown)}")
    if switches.get("reference_mode", "legacy") != "legacy":
        raise ValueError("reference_mode other than legacy is not implemented in this entrypoint")
    for name in UNIMPLEMENTED_SWITCHES:
        if bool(switches.get(name, False)):
            raise ValueError(f"experimental switch {name} is not implemented")
    if variant not in IMPLEMENTED_MODES:
        raise ValueError(f"unknown predictor variant: {variant}")
    warnings = []
    required = []
    if variant in {"analytic_cv", "motion_only"}:
        dependency = "motion/context + supervision manifest"
    elif variant == "constant_legacy":
        dependency = "legacy scene cache or explicit zero scene control"
        required = list(scene_reports)
    elif variant == "geometry_only":
        dependency = "scene_xyz + scene_mask geometry cache"
        required = list(scene_reports)
    elif variant == "geometry_utonia":
        dependency = "matching observed geometry + Utonia feature cache"
        required = list(scene_reports)
    elif variant == "oracle_partial_visible_geometry":
        dependency = "privileged partial visible geometry cache"
        required = list(scene_reports)
    else:
        dependency = "privileged visible geometry cache"
        required = list(scene_reports)
    for report_path in required:
        report = _read(report_path)
        schema = report.get("cache_schema", report.get("schema"))
        if protocol == "formal":
            accepted_formal_schemas = {"observed_utonia_tokens_v2"}
            if variant == "geometry_only":
                accepted_formal_schemas |= {"geometry_only_observed_v1"}
            if schema not in accepted_formal_schemas:
                raise ValueError(f"formal provenance gate failed for {report_path}: {schema}")
            required_fields = {"output_sha256", "input_pixels_sha256", "observed_indices", "feature_shape"}
            missing = sorted(field for field in required_fields if field not in report)
            if missing:
                raise ValueError(f"formal provenance fields missing for {report_path}: {missing}")
            if report.get("privileged_geometry"):
                raise ValueError("formal visual cache cannot be a privileged geometry cache")
            if report.get("static_scene_gt_used") or report.get("future_state_used"):
                raise ValueError(f"formal cache contains privileged/future data: {report_path}")
        else:
            if schema == "observed_utonia_tokens_v2":
                pass
            elif allow_legacy and schema in {None, "observed_utonia_tokens_v1"}:
                warnings.append(f"diagnostic legacy/unverified cache allowed: {report_path}")
            elif variant == "oracle_partial_visible_geometry" and report.get("privileged_geometry"):
                warnings.append(f"diagnostic privileged partial oracle: {report_path}")
            else:
                raise ValueError(f"diagnostic dependency missing or mismatched: {report_path}")
    if protocol == "formal" and not bool(config.get("training", {}).get("formal_training", False)):
        raise ValueError("formal entrypoint requires training.formal_training=true")
    return {"status": "PASS", "protocol": protocol, "variant": variant,
            "dependency": dependency, "warnings": warnings,
            "required_reports": [str(x) for x in required],
            "dit": bool(config.get("training", {}).get("dit", False))}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", choices=("diagnostic", "formal"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--scene-report", type=Path, action="append", default=[])
    parser.add_argument("--allow-legacy", action="store_true")
    args = parser.parse_args()
    result = validate(args.config, protocol=args.protocol, variant=args.variant,
                      scene_reports=args.scene_report, allow_legacy=args.allow_legacy)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
