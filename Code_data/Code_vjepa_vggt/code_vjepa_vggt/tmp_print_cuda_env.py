import os

import torch


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "-1"))
    print(
        {
            "rank": rank,
            "local_rank": local_rank,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
