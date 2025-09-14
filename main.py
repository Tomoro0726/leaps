import os

os.environ["USE_TORCH"] = "ON"

from typing import List

from src.config import Config
from src.runner import Runner


def main():
    cfg = Config()

    runner = Runner(cfg)

    sequences: List[str] = runner.run()
    print(sequences)


if __name__ == "__main__":
    main()
