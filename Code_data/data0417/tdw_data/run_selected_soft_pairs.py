from __future__ import annotations

import sys

sys.path.insert(0, "/home/gaoya/Code_Video/Code_data/data0417/tdw_data")

import run_tdw_genesis_format_exports as m


WANTED = {
    "soft_volume_pair_rubber_sphere_hits_canvas_sphere_site_fast",
    "soft_volume_pair_plastic_sphere_hits_cotton_sphere_suburb_left",
    "soft_volume_pair_silk_sphere_hits_wool_sphere_craftroom_diag",
    "soft_volume_pair_burlap_sphere_hits_rubber_sphere_craftroom_slow",
}


def main() -> None:
    m.CASES = [case for case in m.CASES if case["case_name"] in WANTED]
    for case in m.CASES:
        print(m.record_case(case), flush=True)


if __name__ == "__main__":
    main()
