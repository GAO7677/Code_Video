from __future__ import annotations

import sys

sys.path.insert(0, "/home/gaoya/Code_Video/Code_data/data0417/tdw_data")

import run_tdw_genesis_format_exports as m


WANTED = {
    "soft_volume_pair_rubber_cube_hits_canvas_sphere_site",
    "soft_volume_pair_plastic_cube_hits_cotton_sphere_suburb",
}


def main() -> None:
    m.CASES = [case for case in m.CASES if case["case_name"] in WANTED]
    for case in m.CASES:
        print(m.record_case(case), flush=True)


if __name__ == "__main__":
    main()
