from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from tqdm import trange
from transformers import AutoTokenizer, EsmModel

from src.state import State


class EarlyStopper:
    """
    早期終了を行うクラス
    """

    def __init__(self, cfg: Mapping[str, Any], state: State) -> None:
        """
        Args:
            cfg (Mapping[str, Any]): 設定
            state (State): 状態
        """
        cfg = SimpleNamespace(**cfg)
        self.state = state

        seed: int = cfg.seed
        self.debug: bool = cfg.debug
        self.device: torch.device = cfg.device

        project_dir: Path = Path("runs") / cfg.project

        self.figure_dir: Path = project_dir / "early_stopper" / "figure"
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size: int = cfg.batch_size
        self.patience: int = cfg.patience
        self.num_samples: int = cfg.num_samples

        self.pca = PCA(n_components=2, random_state=seed)
        self.xlim: Tuple[float, float] | None = None
        self.ylim: Tuple[float, float] | None = None

        self.model = (
            EsmModel.from_pretrained(cfg.model_name_or_path).to(self.device).eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

        self.count = 0
        self.best_score = None

        random.seed(seed)

    @torch.inference_mode()
    def _embed(self, sequences: List[str]) -> torch.Tensor:
        """
        Args:
            sequences (List[str]): タンパク質のリスト

        Returns:
            torch.Tensor: 埋め込みベクトル
        """
        results: List[torch.Tensor] = []

        for i in trange(
            0,
            len(sequences),
            self.batch_size,
            desc="Embedding sequences",
            disable=not self.debug,
        ):
            batch = sequences[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            last_hidden_state = outputs["last_hidden_state"]
            attention_mask = inputs["attention_mask"]
            attention_mask = attention_mask.unsqueeze(-1)
            pooled_output = (last_hidden_state * attention_mask).sum(
                1
            ) / attention_mask.sum(1)
            results.append(pooled_output)

        results = torch.cat(results, dim=0)

        return results

    def _plot(
        self, inputs: np.ndarray, targets: np.ndarray, save_path: str | Path
    ) -> None:
        """
        Args:
            inputs (np.ndarray): 入力のベクトル
            targets (np.ndarray): 目的のベクトル
            save_path (str | Path): 保存先のパス
        """
        X = np.concatenate([inputs, targets], axis=0)
        y = np.array([0] * len(inputs) + [1] * len(targets))

        try:
            check_is_fitted(self.pca)
            X_pca = self.pca.transform(X)
        except NotFittedError:
            X_pca = self.pca.fit_transform(X)

        if self.xlim is None:
            left, right = X_pca[:, 0].min(), X_pca[:, 0].max()
            self.xlim = (left, right)

        if self.ylim is None:
            bottom, top = X_pca[:, 1].min(), X_pca[:, 1].max()
            self.ylim = (bottom, top)

        plt.figure(figsize=(8, 8), tight_layout=True)
        for i, label in enumerate(["prev", "next"]):
            plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], lw=2, label=label)
        plt.legend()
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.savefig(save_path)
        plt.close()

    def _mse(self, input: torch.Tensor, target: torch.Tensor) -> float:
        """
        Args:
            input (torch.Tensor): 入力のテンソル
            target (torch.Tensor): 目的のテンソル

        Returns:
            float: 平均二乗誤差
        """
        #  return F.mse_loss(input.mean(0), target.mean(0), reduction="mean").item()
        return F.mse_loss(input, target, reduction="mean").item()

    def __call__(self, prev: List[str], next: List[str]) -> bool:
        """
        Args:
            prev (List[str]): 前のタンパク質のリスト
            next (List[str]): 次のタンパク質のリスト

        Returns:
            bool: 判定した結果
        """
        prev = random.sample(prev, k=self.num_samples)
        next = random.sample(next, k=self.num_samples)

        inputs = self._embed(prev)
        targets = self._embed(next)

        output = self._mse(inputs, targets)

        inputs = inputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        save_path = self.figure_dir / f"iter{self.state.iteration}.png"
        self._plot(inputs, targets, save_path)

        score = 1.0 / (output + 1e-8)

        if self.best_score is None:
            self.best_score = score
            self.count = 0

            return False

        if score > self.best_score:
            self.best_score = score
            self.count = 0
        else:
            self.count += 1

        if self.count >= self.patience:
            return True

        return False
