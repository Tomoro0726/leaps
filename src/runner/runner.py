from typing import List

from src.config import Config
from src.early_stopper import EarlyStopper
from src.evaluator import Evaluator
from src.generator import Generator
from src.predictor import Predictor
from src.sampler import Sampler


class Runner:
    """
    実行を管理するクラス
    """

    def __init__(self, cfg: Config) -> None:
        """
        Args:
            cfg (Mapping[str, object]): 設定
        """
        self.cfg = cfg

        self.num_iterations: int = cfg["runner"]["num_iterations"]
        self.num_sequences: int = cfg["runner"]["num_sequences"]

        self.sampler = Sampler(cfg["sampler"])

        self.predictors = {
            k: Predictor(v) for k, v in cfg["predictor"].items() if isinstance(v, dict)
        }

        self.evaluator = Evaluator(cfg["evaluator"], predictors=self.predictors)
        self.generator = Generator(cfg["generator"])
        self.early_stopper = EarlyStopper(cfg["early_stopper"])

    def run(self) -> List[str]:
        outputs: List[str] | List[List[str]] = self.sampler.sample()
        # outputs: List[str] | List[List[str]] = self.sampler.load()

        for _, predictor in self.predictors.items():
            predictor.train()
            # predictor.load()

        prev = None
        next = outputs

        for iteration in range(1, self.num_iterations + 1):
            outputs = self.evaluator.filter(iteration, next)

            self.generator.train(iteration, outputs)
            self.generator.tune()

            next = self.generator.generate(self.num_sequences)

            if prev is not None and self.early_stopper(prev, next):
                break

            prev = next

        return next
