import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import torch
from tqdm import trange
from transformers import AutoTokenizer, EsmForMaskedLM

from src.evaluator.evaluator import Evaluator
from src.state.state import State


class Likelihood(Evaluator):
    """
    対数尤度でスクリーニングするクラス
    """

    def __init__(self, cfg: Mapping[str, Any], state: State) -> None:
        """
        Args:
            cfg (Mapping[str, Any]): 設定
        """
        cfg = SimpleNamespace(**cfg)
        self.state = state

        self.debug: bool = cfg.debug
        self.device: torch.device = cfg.device

        project_dir: Path = Path("runs") / cfg.project

        self.figure_dir: Path = project_dir / "evaluator" / "likelihood" / "figure"
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.result_dir: Path = project_dir / "evaluator" / "likelihood" / "result"
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.structure_dir: Path = project_dir / "sampler" / "structure"

        self.batch_size: int = cfg.batch_size

        self.mode: Optional[str] = getattr(cfg, "mode", None)
        self.lower: Optional[float] = getattr(cfg, "lower", None)
        self.upper: Optional[float] = getattr(cfg, "upper", None)

        self.threshold: Optional[float] = getattr(cfg, "threshold", None)
        self.top_p: Optional[float] = getattr(cfg, "top_p", None)
        self.top_k: Optional[int] = getattr(cfg, "top_k", None)

        self.model = (
            EsmForMaskedLM.from_pretrained(cfg.model_name_or_path)
            .to(self.device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

    def _get_seq3di(
        self,
        pdb_path: Path,
        chains: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Args:
            pdb_path (Path): PDBファイルのパス
            chains (Optional[List[str]]): 取得したいチェインのリスト

        Returns:
            Dict[str, str]: タンパク質の構造の辞書
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

            results: Dict[str, str] = {}

            with output_path.open() as f:
                for line in f:
                    header, _, seq3di = line.strip().split("\t")[:3]
                    chain = (
                        header.split(" ")[0].replace(pdb_path.name, "").split("_")[-1]
                    )

                    if chain in (chains or ["A"]):
                        results[chain] = seq3di

            return results

    @torch.inference_mode()
    def _log_likelihood(
        self, wt_sequence: str, seq3di: str, sequences: List[str]
    ) -> List[float]:
        """
        Args:
            wt_sequence (str): 野生型のタンパク質
            seq3di (str): 野生型のタンパク質の構造
            sequences (List[str]): タンパク質のリスト

        Returns:
            List[float]: 各タンパク質の対数尤度
        """
        vocab = "pynwrqhgdlvtmfsaeikc#"

        lls: List[float] = []

        for seq in sequences:
            masked_sequences = []
            positions = []

            indices = [
                i for i, (aa, bb) in enumerate(zip(wt_sequence, seq)) if aa != bb
            ]

            for i in indices:
                tokens = [aa + seq3di[j] for j, aa in enumerate(wt_sequence)]
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

    def _score(self, sequences: List[str], pdb_path: Path) -> List[float]:
        """
        Args:
            sequences (List[str]): タンパク質のリスト
            pdb_path (Path): PDBファイルのパス

        Returns:
            List[float]: 各タンパク質の対数尤度
        """
        seq3di = self._get_seq3di(pdb_path)["A"]

        scores: List[float] = []

        for i in trange(
            0,
            len(sequences),
            self.batch_size,
            desc="Computing likelihoods",
            disable=not self.debug,
        ):
            batch = sequences[i : i + self.batch_size]
            outputs = self._log_likelihood(sequences[0], seq3di, batch)
            scores.extend(outputs)

        return scores

    def filter(self, samples: Dict[str, List[str]]) -> List[str]:
        """
        Args:
            samples (Dict[str, List[str]]): サンプリングされたタンパク質の辞書

        Returns:
            List[str]: スクリーニングされたタンパク質のリスト
        """
        sequences: List[str] = []
        scores: List[float] = []

        for id, sample in samples.items():
            pdb_path = next(self.structure_dir.glob(f"{id}_unrelaxed_rank_001_*.pdb"))
            sequences.extend(sample)
            scores.extend(self._score(sample, pdb_path))

        save_path = self.result_dir / f"iter{self.state.iteration}.csv"
        self._save(sequences, scores, save_path)

        save_path = self.figure_dir / f"iter{self.state.iteration}.png"
        self._plot(scores, save_path)

        if self.threshold is not None:
            if self.mode == "max":
                return [
                    seq for seq, sc in zip(sequences, scores) if sc >= self.threshold
                ]

            if self.mode == "min":
                return [
                    seq for seq, sc in zip(sequences, scores) if sc <= self.threshold
                ]

        results = self._sort(sequences, scores)

        if self.top_p is not None:
            index = int(len(results) * self.top_p)
            return [seq for seq, _ in results[:index]]

        if self.top_k is not None:
            index = self.top_k
            return [seq for seq, _ in results[:index]]

        return []
