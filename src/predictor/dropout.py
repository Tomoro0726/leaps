import torch
import torch.nn as nn


class Dropout(nn.Module):
    """
    ドロップアウトを定義するクラス
    """

    def __init__(self, p: float = 0.5, eps: float = 1e-8):
        super().__init__()

        assert 0.0 <= p < 1.0

        self.p: float = p
        self.eps: float = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 入力のテンソル

        Returns:
            torch.Tensor: ドロップアウトしたテンソル
        """
        if not self.training:
            return x

        dim = x.size(-1)
        k = max(1, int(round(dim * self.p)))

        flat = x.reshape(-1, dim)

        with torch.no_grad():
            std = flat.std(dim=0, unbiased=False)
            med = flat.median(dim=0).values
            cvar = std / (med.abs() + self.eps)

            _, index = torch.topk(cvar, k, largest=True)
            mask = torch.ones(dim, device=x.device, dtype=x.dtype)
            mask[index] = 0

            scale = 1.0 / (1.0 - k / float(dim))

        shape = (1,) * (x.dim() - 1) + (dim,)
        return x * mask.view(shape) * scale
