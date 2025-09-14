from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.predictor.dropout import Dropout


class Regressor(nn.Module):
    """
    回帰モデルを定義するクラス
    """

    def __init__(self, backbone: AutoModel, hidden_dim: int) -> None:
        """
        Args:
            backbone (AutoModel): 学習済みモデル
            hidden_dim (int): 隠れ層の次元数
        """

        super().__init__()

        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.head = nn.Sequential(
            Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids (torch.Tensor): トークン
            attention_mask (torch.Tensor): マスク
            labels (torch.Tensor): ラベル

        Returns:
            SequenceClassifierOutput: 結果
        """
        output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        last_hidden_state = output.last_hidden_state

        extended_attention_mask = attention_mask.to(
            dtype=last_hidden_state.dtype
        ).unsqueeze(-1)
        pooled_output = (last_hidden_state * extended_attention_mask).sum(
            1
        ) / extended_attention_mask.sum(1).clamp(min=1e-9)

        logits = self.head(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
