import random
from types import SimpleNamespace
from typing import List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import AutoTokenizer, EsmModel


class EarlyStopper:
    """
    早期終了を行うクラス
    """

    def __init__(self, cfg: Mapping[str, object]):
        """
        Args:
            cfg (Mapping[str, object]): 設定
        """
        self.cfg = SimpleNamespace(**cfg)

        self.seed: int = self.cfg.seed
        self.debug: bool = self.cfg.debug
        self.device: torch.device = self.cfg.device

        self.batch_size: int = self.cfg.batch_size
        self.patience: int = self.cfg.patience
        self.num_samples: int = self.cfg.num_samples

        np.random.seed(self.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name_or_path)
        self.model = (
            EsmModel.from_pretrained(self.cfg.model_name_or_path).to(self.device).eval()
        )

        self.count = 0
        self.best_score = None

    @torch.inference_mode()
    def _embed(self, sequences: List[str]) -> torch.Tensor:
        """
        Args:
            sequences (List[str]): タンパク質のリスト

        Returns:
            torch.Tensor: 埋め込みベクトル
        """
        results: List[torch.Tensor] = []

        for i in trange(0, len(sequences), self.batch_size, disable=not self.debug):
            batch = sequences[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            extended_attention_mask = attention_mask.unsqueeze(-1)
            pooled_outputs = (last_hidden_state * extended_attention_mask).sum(
                1
            ) / extended_attention_mask.sum(1)
            results.append(pooled_outputs.detach().cpu())

        results = torch.cat(results, dim=0)

        return results

    def _mse(self, input: torch.Tensor, target: torch.Tensor) -> float:
        """
        Args:
            input (torch.Tensor): 入力のテンソル
            target (torch.Tensor): 目的のテンソル

        Returns:
            float: 平均二乗誤差
        """
        return F.mse_loss(input, target, reduction="mean").item()

    def __call__(self, prev: Sequence[str], next: Sequence[str]) -> bool:
        """
        Args:
            prev (Sequence[str]): 前のタンパク質のリスト
            next (Sequence[str]): 次のタンパク質のリスト
        Returns:
            bool: 判定した結果
        """
        prev = random.choices(prev, k=self.num_samples)
        next = random.choices(next, k=self.num_samples)

        inputs = self._embed(prev)
        targets = self._embed(next)

        output = self._mse(inputs, targets)

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
