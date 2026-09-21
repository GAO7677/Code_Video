"""Prepare the authorized full-432 replay from a frozen history table.

The wrapper is intentionally a separate entry point from the executed pilot.
It validates a predeclared 48-history control table, installs it into the
versioned pilot generator, and then delegates to the same Bullet replay and
manifest writer.  It is IMPLEMENTED_NOT_RUN in this phase.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import prepare_physvideo_phase1_pilot as pilot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-seed", type=int, default=2026092111)
    args = parser.parse_args()
    table_payload = json.loads(args.controls_json.read_text())
    table = table_payload.get("families", table_payload)
    expected = set(pilot.FAMILY_CONFIG)
    if set(table) != expected:
        raise ValueError(f"controls must contain exactly {sorted(expected)}")
    controls = {}
    for family in expected:
        rows = table[family]
        if len(rows) != 48:
            raise ValueError(f"{family} requires 48 predeclared history rows")
        required = {"speed_mps", "heading_deg", "offset_y_m", "start_x_m"}
        for index, row in enumerate(rows):
            if set(row) != required or not all(isinstance(row[key], (int, float)) for key in required):
                raise ValueError(f"invalid {family} control row {index}")
        controls[family] = tuple({key: float(value) for key, value in row.items()} for row in rows)
    # The delegated generator refuses an existing output and records its own
    # source hashes.  Keep its CLI surface unchanged so the executed pilot
    # remains reproducible.
    pilot.FAMILY_HISTORY_CONTROLS = controls
    pilot.PROTOCOL_VERSION = "physvideo_next_experiment_protocol_20260921_full432_v1"
    sys.argv = [sys.argv[0], "--output", str(args.output), "--point-seed", str(args.point_seed)]
    pilot.main()


if __name__ == "__main__":
    main()
