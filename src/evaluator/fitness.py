from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from tqdm import trange

from src.predictor.predictor import Predictor


class Fitness:
    """
    機能でスクリーニングするクラス
    """

    def __init__(self, cfg: Mapping[str, object], predictor: Predictor) -> None:
        """
        Args:
            cfg (Mapping[str, object]): 設定
            predictor (Predictor): 予測器
        """
        self.cfg = SimpleNamespace(**cfg)

        self.debug: bool = self.cfg.debug
        self.device: torch.device = self.cfg.device

        self.batch_size: int = self.cfg.batch_size

        self.mode: Optional[str] = getattr(self.cfg, "mode", None)
        self.lower: Optional[float] = getattr(self.cfg, "lower", None)
        self.upper: Optional[float] = getattr(self.cfg, "upper", None)

        self.series: Dict[str, Any] = self.cfg.series
        self.parallel: Dict[str, Any] = self.cfg.parallel

        self.predictor = predictor

    @torch.inference_mode()
    def _predict(self, sequences: Sequence[str]) -> List[float]:
        """
        Args:
            sequences (Sequence[str]): タンパク質のリスト

        Returns:
            List[float]: 各タンパク質の予測値
        """
        outputs = self.predictor.predict(sequences)
        return outputs

    def _score(self, sequences: Sequence[str]) -> List[float]:
        """
        Args:
            sequences (Sequence[str]): タンパク質のリスト

        Returns:
            List[float]: 各タンパク質の予測値
        """
        scores: List[float] = []

        for i in trange(0, len(sequences), self.batch_size, disable=not self.debug):
            batch = sequences[i : i + self.batch_size]
            outputs = self._predict(batch)
            scores.extend(outputs)

        return scores

    def _sort(
        self,
        sequences: Sequence[str],
        scores: Sequence[float],
    ) -> List[Tuple[str, float]]:
        """
        Args:
            sequences (Sequence[str]): ソートしたいタンパク質のリスト
            scores (Sequence[float]): ソートしたいタンパク質の予測値

        Returns:
            List[Tuple[str, float]]: ソートされたリスト
        """

        if self.mode == "max":
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        elif self.mode == "min":
            indices = sorted(range(len(scores)), key=lambda i: scores[i])

        elif self.mode == "range":
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

        return [(sequences[i], scores[i]) for i in indices]

    def filter(self, sequences: Sequence[str], strategy: str) -> List[str]:
        """
        Args:
            sequences (Sequence[str]): タンパク質のリスト
            strategy (str): スクリーニングの戦略（"series" または "parallel"）

        Returns:
            List[str]: スクリーニングされたタンパク質のリスト
        """
        scores = self._score(sequences)

        assert strategy in ["series", "parallel"]

        if strategy == "series":
            threshold = self.series.get("threshold")
            top_p = self.series.get("top_p")
            top_k = self.series.get("top_k")

        if strategy == "parallel":
            threshold = self.parallel.get("threshold")
            top_p = self.parallel.get("top_p")
            top_k = self.parallel.get("top_k")

        if threshold is not None:
            return [seq for seq, sc in zip(sequences, scores) if sc >= threshold]

        results = self._sort(sequences, scores)

        if top_p is not None:
            index = int(len(results) * top_p)
            return [seq for seq, _ in results[:index]]

        if top_k is not None:
            index = top_k
            return [seq for seq, _ in results[:index]]

        return []
