from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage3 bridge finetune entry placeholder. The first version keeps Stage3 as a documented scaffold."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-checkpoint", default=None)
    args = parser.parse_args()
    raise NotImplementedError(
        "Stage3 bridge finetune is intentionally left as a scaffold in the first version. "
        "Stage1 oracle injection and Stage2 predictor are the implemented paths."
    )


if __name__ == "__main__":
    main()
