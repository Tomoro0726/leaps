from itertools import chain
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from evcouplings.couplings.model import CouplingsModel

from src.evaluator.evaluator import Evaluator
from src.state.state import State


class Hamiltonian(Evaluator):
    """
    ハミルトニアンでスクリーニングするクラス
    """

    def __init__(self, cfg: Mapping[str, Any], state: State) -> None:
        """
        Args:
            cfg (Mapping[str, Any]): 設定
            state (State): 状態
        """
        cfg = SimpleNamespace(**cfg)
        self.state = state

        self.debug: bool = cfg.debug
        self.device: torch.device = cfg.device

        self.project_dir: Path = Path("runs") / cfg.project

        self.figure_dir: Path = (
            self.project_dir / "evaluator" / "hamiltonian" / "figure"
        )
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.result_dir: Path = (
            self.project_dir / "evaluator" / "hamiltonian" / "result"
        )
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.msa_dir: Path = self.project_dir / "evaluator" / "hamiltonian" / "msa"
        self.msa_dir.mkdir(parents=True, exist_ok=True)

        self.weight_dir: Path = (
            self.project_dir / "evaluator" / "hamiltonian" / "weight"
        )
        self.weight_dir.mkdir(parents=True, exist_ok=True)

        self.mode: Optional[str] = getattr(cfg, "mode", None)
        self.lower: Optional[float] = getattr(cfg, "lower", None)
        self.upper: Optional[float] = getattr(cfg, "upper", None)

        self.threshold: Optional[float] = getattr(cfg, "threshold", None)
        self.top_p: Optional[float] = getattr(cfg, "top_p", None)
        self.top_k: Optional[int] = getattr(cfg, "top_k", None)

        csv_path: Path = self.project_dir / "data" / "input.csv"
        df = pd.read_csv(csv_path)
        self.id: str = df["id"].tolist()[0]
        self.wt_sequence: str = df["sequence"].tolist()[0]

    def _msa(self) -> List[str]:
        """
        Returns:
            List[str]: MSAによるタンパク質のリスト
        """
        fasta_path = self.project_dir / "evaluator" / "hamiltonian" / "query.fasta"

        record = SeqRecord(Seq(self.wt_sequence), id=self.id, description="")
        SeqIO.write([record], fasta_path, "fasta")

        subprocess.run(
            ["colabfold_batch", fasta_path, self.msa_dir, "--msa-only"], check=True
        )

        def normalize(sequence: str) -> str:
            """
            Args:
                sequence (str): 正規化したいタンパク質

            Returns:
                str: 正規化されたタンパク質
            """
            sequence = sequence.replace(".", "-")

            outputs = []

            for aa in sequence:
                if aa.islower():
                    continue
                if aa in IUPACData.protein_letters + "-":
                    outputs.append(aa)
                else:
                    outputs.append("-")

            return "".join(outputs)

        sequecnes: List[str] = []

        header: str | None = None
        buffer: List[str] = []

        a3m_path = self.msa_dir / f"{self.id}.a3m"
        with open(a3m_path) as f:
            for line in chain(f, [">"]):
                line = line.strip()

                if line.startswith("#"):
                    continue

                if line.startswith(">"):
                    if header is not None:
                        sequence = normalize("".join(buffer))
                        if len(sequence) == len(self.wt_sequence):
                            sequecnes.append(sequence)

                    if len(line) == 1:
                        continue

                    header = line[1:].split()[0]
                    buffer = []
                else:
                    buffer.append(line)

        return sequecnes

    def train(self) -> None:
        sequences = self._msa()

        fasta_path = self.project_dir / "evaluator" / "hamiltonian" / "train.fasta"

        records: List[SeqRecord] = []
        for i, seq in enumerate(sequences, start=1):
            record = SeqRecord(Seq(seq), id=str(i), description="")
            records.append(record)

        SeqIO.write(records, fasta_path, "fasta")

        subprocess.run(
            [
                os.path.join("bin", "plmc"),
                "-o",
                self.weight_dir / "model.params",
                fasta_path,
            ],
            check=True,
        )

    def _score(self, sequences: List[str]) -> List[float]:
        """
        Args:
            sequences (List[str]): タンパク質のリスト

        Returns:
            List[float]: 各タンパク質のハミルトニアン
        """
        model = CouplingsModel(self.weight_dir / "model.params", file_format="plmc_v2")

        # e_wt, _, _ = model.hamiltonians([self.wt_sequence])[0]
        # _, e_wt, _ = model.hamiltonians([self.wt_sequence])[0]
        _, _, e_wt = model.hamiltonians([self.wt_sequence])[0]

        energies = model.hamiltonians(sequences)
        # e_mut = np.array([e[0] for e in energies])
        # e_mut = np.array([e[1] for e in energies])
        e_mut = np.array([e[2] for e in energies])

        scores = (e_mut - e_wt).tolist()

        return scores

    def filter(self, sequences: List[str]) -> List[str]:
        """
        Args:
            sequences (List[str]): タンパク質のリスト

        Returns:
            List[str]: スクリーニングされたタンパク質のリスト
        """
        scores = self._score(sequences)

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
