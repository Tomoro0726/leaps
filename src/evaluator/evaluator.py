from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Evaluator(ABC):
    """
    スクリーニングするクラス
    """

    mode: Optional[str]
    lower: Optional[float]
    upper: Optional[float]

    @abstractmethod
    def filter(self, *args, **kwargs) -> List[str]:
        raise NotImplementedError

    def _plot(self, scores: List[float], save_path: str | Path) -> None:
        """
        Args:
            scores (List[float]): スコアのリスト
            save_path (str | Path): 保存先のパス
        """
        save_path = Path(save_path)

        x = np.asarray(scores, dtype=float)

        plt.figure(figsize=(8, 6), tight_layout=True)
        plt.hist(x, bins="fd")
        plt.savefig(save_path)
        plt.close()

    def _save(
        self,
        sequences: List[str],
        scores: List[float],
        save_path: str | Path,
    ) -> None:
        """
        Args:
            sequences (List[str]): 保存したいタンパク質のリスト
            scores (List[float]): 保存したいスコアのリスト
            save_path (str | Path): 保存先のパス
        """
        save_path = Path(save_path)

        df = pd.DataFrame({"sequence": sequences, "score": scores})
        df.to_csv(save_path, index=False)

    def _sort(
        self,
        sequences: List[str],
        scores: List[float],
    ) -> List[Tuple[str, float]]:
        """
        Args:
            sequences (List[str]): ソートしたいタンパク質のリスト
            scores (List[float]): ソートしたいスコアのリスト

        Returns:
            List[Tuple[str, float]]: ソートされたリスト
        """
        if getattr(self, "mode", None) == "max":
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        elif getattr(self, "mode", None) == "min":
            indices = sorted(range(len(scores)), key=lambda i: scores[i])

        elif getattr(self, "mode", None) == "range":
            assert self.lower is not None and self.upper is not None

            center = 0.5 * (self.lower + self.upper)

            def key(i: int):
                score = scores[i]
                if self.lower <= score <= self.upper:
                    return (0.0, abs(score - center))
                dist = (
                    (self.lower - score) if score < self.lower else (score - self.upper)
                )
                return (abs(dist), abs(score - center))

            indices = sorted(range(len(scores)), key=key)

        else:
            indices = []

        outputs = [(sequences[i], scores[i]) for i in indices]

        return outputs
