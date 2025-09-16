from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import torch

from src.evaluator.evaluator import Evaluator
from src.predictor import Predictor
from src.state import State


class Fitness(Evaluator):
    """
    機能でスクリーニングするクラス
    """

    def __init__(
        self, cfg: Mapping[str, Any], state: State, name: str, predictor: Predictor
    ) -> None:
        """
        Args:
            cfg (Mapping[str, Any]): 設定
            state (State): 状態
            name (str): 評価器の名前
            predictor (Predictor): 予測器
        """
        cfg = SimpleNamespace(**cfg)
        self.state = state

        self.debug: bool = cfg.debug
        self.device: torch.device = cfg.device

        project_dir: Path = Path("runs") / cfg.project

        self.figure_dir: Path = project_dir / "evaluator" / "fitness" / name / "figure"
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.result_dir: Path = project_dir / "evaluator" / "fitness" / name / "result"
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size: int = cfg.batch_size

        self.mode: Optional[str] = getattr(cfg, "mode", None)
        self.lower: Optional[float] = getattr(cfg, "lower", None)
        self.upper: Optional[float] = getattr(cfg, "upper", None)

        self.series: Dict[str, Any] = cfg.series
        self.parallel: Dict[str, Any] = cfg.parallel

        self.predictor = predictor

    @torch.inference_mode()
    def _predict(self, sequences: List[str]) -> List[float]:
        """
        Args:
            sequences (List[str]): タンパク質のリスト

        Returns:
            List[float]: 各タンパク質の予測値
        """
        outputs = self.predictor.predict(sequences)

        return outputs

    def _score(self, sequences: List[str]) -> List[float]:
        """
        Args:
            sequences (List[str]): タンパク質のリスト

        Returns:
            List[float]: 各タンパク質の予測値
        """
        scores: List[float] = self._predict(sequences)

        return scores

    def filter(self, sequences: List[str], strategy: str) -> List[str]:
        """
        Args:
            sequences (List[str]): タンパク質のリスト
            strategy (str): スクリーニングの戦略

        Returns:
            List[str]: スクリーニングされたタンパク質のリスト
        """
        assert strategy in ["series", "parallel"]

        scores = self._score(sequences)

        save_dir = self.result_dir / strategy
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"iter{self.state.iteration}.csv"
        self._save(sequences, scores, save_path)

        save_dir = self.figure_dir / strategy
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"iter{self.state.iteration}.png"
        self._plot(scores, save_path)

        if strategy == "series":
            threshold = self.series.get("threshold")
            top_p = self.series.get("top_p")
            top_k = self.series.get("top_k")

        if strategy == "parallel":
            threshold = self.parallel.get("threshold")
            top_p = self.parallel.get("top_p")
            top_k = self.parallel.get("top_k")

        if threshold is not None:
            if self.mode == "max":
                return [seq for seq, sc in zip(sequences, scores) if sc >= threshold]

            if self.mode == "min":
                return [seq for seq, sc in zip(sequences, scores) if sc <= threshold]

        results = self._sort(sequences, scores)

        if top_p is not None:
            index = int(len(results) * top_p)
            return [seq for seq, _ in results[:index]]

        if top_k is not None:
            index = top_k
            return [seq for seq, _ in results[:index]]

        return []
