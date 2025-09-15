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

    def __init__(self, model: AutoModel, hidden_dim: int) -> None:
        """
        Args:
            model (AutoModel): 学習済みモデル
            hidden_dim (int): 隠れ層の次元数
        """

        super().__init__()

        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.regressor = nn.Sequential(
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
        **_,
    ) -> SequenceClassifierOutput:
        """
        Args:
            input_ids (torch.Tensor): トークン
            attention_mask (torch.Tensor): マスク
            labels (torch.Tensor): ラベル

        Returns:
            SequenceClassifierOutput: 結果
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs["last_hidden_state"]
        attention_mask = attention_mask.unsqueeze(-1)
        pooled_output = (last_hidden_state * attention_mask).sum(
            1
        ) / attention_mask.sum(1)

        logits = self.regressor(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
