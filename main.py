import os

os.environ["USE_TORCH"] = "ON"

from typing import List

from src.config import Config
from src.runner import Runner


def main():
    cfg = Config()

    csv_path = os.path.join("data", "gfp.csv")
    runner = Runner(cfg, csv_path)

    sequences: List[str] = runner.run()
    print(sequences)


if __name__ == "__main__":
    main()
