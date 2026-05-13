from __future__ import annotations

import sys

sys.path.insert(0, "/home/gaoya/Code_Video/Code_data/data0417/tdw_data")

import run_tdw_genesis_format_exports as m


WANTED = {
    "cloth_drop_wicker_then_toaster_hit_suburb",
    "cloth_drop_cardboard_then_box_hit_site",
    "rigid_pair_bowl_hits_shoebox_suburb",
    "rigid_pair_camera_box_hits_toaster_site",
}


def main() -> None:
    m.CASES = [case for case in m.CASES if case["case_name"] in WANTED]
    for case in m.CASES:
        print(m.record_case(case), flush=True)


if __name__ == "__main__":
    main()
