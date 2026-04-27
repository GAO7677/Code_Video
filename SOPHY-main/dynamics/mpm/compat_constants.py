import os
import json
from util.constants import METADATA

MAX_E = 9


with open(os.path.join(METADATA, 'parts_fine.json'), "r") as f:
    parts_fine = json.load(f)

with open(os.path.join(METADATA, 'used_parts_fine.json'), "r") as f:
    used_parts_fine = json.load(f)

PART_I2N = {k: v for k, v in enumerate(parts_fine)}
PART_N2I = {v: k for k, v in enumerate(parts_fine)}

USED_PART_I2N = {k: v for k, v in enumerate(used_parts_fine)}

MAT_PARAMS_DICT = {
    # Ceramic
    0: {
        "name": "ceramic",
        "elasticity": "neo_hookean",
        "plasticity": "von_mises_with_damage",
        "rho":  (2300, 2700),  # kg/m³
        "E": (70e7, 120e7),  # Young's modulus scaled to MPa
        "nu": (0.18, 0.25),  # Poisson's ratio stays the same
        "sigma_y": (100e4, 300e4)  # Yield stress scaled to kPa
    },

    # Fabric
    1: {
        "name": "fabric",
        "elasticity": "fixed_corotated",
        "plasticity": "identity",
        "rho": (50, 200),  # kg/m³
        "E": (1e4, 10e4),  # Young's modulus scaled to kPa
        "nu": (0.3, 0.4)
    },

    # Glass
    2: {
        "name": "glass",
        "elasticity": "neo_hookean",
        "plasticity": "von_mises",
        "rho": (2400, 2600),  # kg/m³
        "E": (50e7, 100e7),  # Young's modulus scaled to MPa
        "nu": (0.2, 0.3),
        "sigma_y": (200e4, 400e4)  # Yield stress scaled to kPa
    },

    # Granite
    3: {
        "name": "granite",
        "elasticity": "stvk",
        "plasticity": "drucker_prager",
        "rho": (2600, 2800),  # kg/m³
        "E": (30e7, 70e7),  # Young's modulus scaled to MPa
        "nu": (0.15, 0.25),
        "phi": (30, 40)  # degrees
    },

    # Leather
    4: {
        "name": "leather",
        "elasticity": "fixed_corotated",
        "plasticity": "identity",
        "rho": (900, 1100),  # kg/m³
        "E": (5e4, 30e4),  # Young's modulus scaled to kPa
        "nu": (0.3, 0.4)
    },

    # Marble
    5: {
        "name": "marble",
        "elasticity": "stvk",
        "plasticity": "drucker_prager",
        "rho": (2600, 2700),  # kg/m³
        "E": (50e7, 70e7),  # Young's modulus scaled to MPa
        "nu": (0.2, 0.3),
        "phi": (30, 40)  # degrees
    },

    # Metal
    6: {
        "name": "metal",
        "elasticity": "neo_hookean",
        "plasticity": "von_mises",
        "rho": (7700, 8900),  # kg/m³
        "E": (100e7, 200e7),  # Young's modulus scaled to MPa
        "nu": (0.3, 0.35),
        "sigma_y": (200e4, 800e4)  # Yield stress scaled to kPa
    },

    # Plant
    7: {
        "name": "plant",
        "elasticity": "fixed_corotated",
        "plasticity": "identity",
        "rho": (500, 1200),  # kg/m³
        "E": (1e4, 5e4),  # Young's modulus scaled to kPa
        "nu": (0.3, 0.4)
    },

    # Plastic
    8: {
        "name": "plastic",
        "elasticity": "neo_hookean",
        "plasticity": "von_mises_with_damage",
        "rho": (1000, 1500),  # kg/m³
        "E": (3e7, 5e7),  # Young's modulus scaled to MPa
        "nu": (0.35, 0.4),
        "sigma_y": (100e4, 150e4)  # Yield stress scaled to kPa
    },

    # Rubber
    9: {
        "name": "rubber",
        "elasticity": "neo_hookean",
        "plasticity": "identity",
        "rho": (1100, 1200),  # kg/m³
        "E": (1e4, 5e4),  # Young's modulus scaled to kPa
        "nu": (0.45, 0.49)
    },

    # Soil
    10: {
        "name": "soil",
        "elasticity": "stvk",
        "plasticity": "drucker_prager",
        "rho": (1500, 2000),  # kg/m³
        "E": (1e4, 10e4),  # Young's modulus scaled to kPa
        "nu": (0.3, 0.4),
        "phi": (25, 35)  # degrees
    },

    # Wax
    11: {
        "name": "wax",
        "elasticity": "neo_hookean",
        "plasticity": "von_mises_with_damage",
        "rho": (900, 950),  # kg/m³
        "E": (50e4, 100e4),  # Young's modulus scaled to kPa
        "nu": (0.3, 0.4),
        "sigma_y": (1e4, 5e4)  # Yield stress scaled to kPa
    },

    # Wood
    12: {
        "name": "wood",
        "elasticity": "stvk",
        "plasticity": "von_mises_with_damage",
        "rho": (600, 800),  # kg/m³
        "E": (10e7, 16e7),  # Young's modulus scaled to MPa
        "nu": (0.2, 0.3),
        "sigma_y": (40e4, 100e4)  # Yield stress scaled to kPa
    }
}

COMB_ID = {
    0: ("neo_hookean", "identity"),
    1: ("neo_hookean", "von_mises_with_damage"),
    2: ("neo_hookean", "von_mises"),
    3: ("stvk", "drucker_prager"),
}
