import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from tqdm import trange
from transformers import AutoTokenizer, EsmForMaskedLM


class Likelihood:
    """
    対数尤度でスクリーニングするクラス
    """

    def __init__(self, cfg: Mapping[str, object]) -> None:
        """
        Args:
            cfg (Mapping[str, object]): 設定
        """
        self.cfg = SimpleNamespace(**cfg)

        self.debug: bool = self.cfg.debug
        self.device: torch.device = self.cfg.device

        self.batch_size: int = self.cfg.batch_size

        self.mode: Optional[str] = getattr(self.cfg, "mode", None)
        self.lower: Optional[float] = getattr(self.cfg, "lower", None)
        self.upper: Optional[float] = getattr(self.cfg, "upper", None)

        self.threshold: Optional[float] = getattr(self.cfg, "threshold", None)
        self.top_p: Optional[float] = getattr(self.cfg, "top_p", None)
        self.top_k: Optional[int] = getattr(self.cfg, "top_k", None)

        self.model = (
            EsmForMaskedLM.from_pretrained(self.cfg.model_name_or_path)
            .to(self.device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name_or_path)

    def _get_structure(
        self,
        pdb_path: Path,
        chains: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[str, str]]:
        """
        Args:
            pdb_path (Path): PDBファイルのパス
            chains (Optional[List[str]]): 取得したいチェインのリスト

        Returns:
            Dict[str, Tuple[str, str]]: タンパク質の構造の辞書
        """
        with tempfile.TemporaryDirectory() as fp:
            output_path = Path(fp) / "output"

            command = [
                os.path.join("bin", "foldseek"),
                "structureto3didescriptor",
                "-v",
                "0",
                "--threads",
                "1",
                "--chain-name-mode",
                "1",
                str(pdb_path),
                str(output_path),
            ]
            subprocess.run(command, check=True)

            results: Dict[str, Tuple[str, str]] = {}

            with output_path.open() as f:
                for line in f:
                    desc, sequence, structure = line.strip().split("\t")[:3]
                    chain = desc.split(" ")[0].replace(pdb_path.name, "").split("_")[-1]

                    if chain in (chains or ["A"]):
                        results[chain] = (sequence, structure)

            return results

    @torch.inference_mode()
    def _log_likelihood(self, sequences: Sequence[str]) -> List[float]:
        """
        Args:
            sequences (Sequence[str]): タンパク質のリスト

        Returns:
            List[float]: 各タンパク質の対数尤度
        """
        vocab = "pynwrqhgdlvtmfsaeikc#"

        lls: List[float] = []

        wt_sequence, structure = self._get_structure(self.pdb_path)["A"]

        for seq in sequences:
            masked_sequences = []
            positions = []

            indices = [
                i for i, (aa, bb) in enumerate(zip(wt_sequence, seq)) if aa != bb
            ]

            for i in indices:
                tokens = [aa + structure[j] for j, aa in enumerate(wt_sequence)]
                tokens[i] = "#" + tokens[i][-1]
                masked_sequences.append(" ".join(tokens))
                positions.append(i + 1)

            ll = 0.0
            for j in range(0, len(masked_sequences), self.batch_size):
                batch = masked_sequences[j : j + self.batch_size]
                inputs = self.tokenizer.batch_encode_plus(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)

                logits = self.model(**inputs).logits
                probs = logits.softmax(dim=-1)

                for k, pos in enumerate(positions[j : j + self.batch_size]):
                    aa = wt_sequence[pos - 1]
                    bb = seq[pos - 1]

                    aa = self.tokenizer.get_vocab()[aa + vocab[0]]
                    bb = self.tokenizer.get_vocab()[bb + vocab[0]]

                    p_wt = probs[k, pos, aa : aa + len(vocab)].sum()
                    p_mt = probs[k, pos, bb : bb + len(vocab)].sum()

                    ll += (torch.log(p_mt + 1e-12) - torch.log(p_wt + 1e-12)).item()

            lls.append(ll)

        return lls

    def _score(self, sequences: Sequence[str]) -> List[float]:
        """
        Args:
            sequences (Sequence[str]): タンパク質のリスト

        Returns:
            List[float]: 各タンパク質の対数尤度
        """
        scores: List[float] = []

        for i in trange(0, len(sequences), self.batch_size, disable=not self.debug):
            batch = sequences[i : i + self.batch_size]
            outputs = self._log_likelihood(batch)
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

    def filter(self, sequences: Sequence[str], pdb_path: Path) -> List[str]:
        """
        Args:
            sequences (Sequence[str]): タンパク質のリスト
            pdb_path (Path): PDBファイルのパス

        Returns:
            List[str]: スクリーニングされたタンパク質のリスト
        """
        self.pdb_path = pdb_path
        scores = self._score(sequences)

        if self.threshold is not None:
            return [seq for seq, sc in zip(sequences, scores) if sc >= self.threshold]

        results = self._sort(sequences, scores)

        if self.top_p is not None:
            index = int(len(results) * self.top_p)
            return [seq for seq, _ in results[:index]]

        if self.top_k is not None:
            index = self.top_k
            return [seq for seq, _ in results[:index]]

        return []
