import random
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping

import pandas as pd
import torch
from Bio import SeqIO
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class Sampler:
    """
    タンパク質をサンプリングするクラス
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        """
        Args:
            cfg (Mapping[str, Any]): 設定
        """
        cfg: SimpleNamespace = SimpleNamespace(**cfg)

        self.seed: int = cfg.seed
        self.device: torch.device = cfg.device

        self.project_dir: Path = Path("runs") / cfg.project

        self.sequence_dir: Path = self.project_dir / "sampler" / "sequence"
        self.sequence_dir.mkdir(parents=True, exist_ok=True)

        self.structure_dir: Path = self.project_dir / "sampler" / "structure"
        self.structure_dir.mkdir(parents=True, exist_ok=True)

        self.num_shuffles: int = cfg.num_shuffles
        self.window_sizes: List[int] = cfg.window_sizes
        self.shuffle_rate: float = cfg.shuffle_rate

        csv_path: Path = self.project_dir / "data" / "input.csv"
        df = pd.read_csv(csv_path)
        self.ids: List[str] = df["id"].tolist()
        self.wt_sequences: List[str] = df["sequence"].tolist()

        random.seed(self.seed)

    def _mutate(self, sequence: str) -> List[str]:
        """
        Args:
            sequence (str): 変異させるタンパク質

        Returns:
            List[str]: 変異されたタンパク質のリスト
        """
        sequence = list(sequence)
        mutant_sequences: List[str] = []

        for i, aa in enumerate(sequence):
            for bb in IUPACData.protein_letters:
                if aa == bb:
                    continue

                mutant_sequence = sequence.copy()
                mutant_sequence[i] = bb
                mutant_sequences.append("".join(mutant_sequence))

        return mutant_sequences

    def _shuffle(self, sequences: List[str]) -> List[str]:
        """
        Args:
            sequences (List[str]): シャッフルするタンパク質のリスト

        Returns:
            List[str]: シャッフルされたタンパク質のリスト
        """
        shuffled_sequences: List[str] = []

        while True:
            for window_size in self.window_sizes:
                num_windows = max(len(seq) // window_size for seq in sequences)

                windows = []

                for seq in sequences:
                    window = []
                    for i in range(len(seq) // window_size):
                        start = i * window_size
                        end = (i + 1) * window_size
                        window.append(seq[start:end])
                    windows.append(window)

                for j in range(num_windows):
                    rows = [i for i, window in enumerate(windows) if j < len(window)]
                    indices = [
                        row for row in rows if random.random() < self.shuffle_rate
                    ]
                    results = [windows[i][j] for i in indices]
                    random.shuffle(results)
                    for i, result in zip(indices, results):
                        windows[i][j] = result

                for i, window in enumerate(windows):
                    sequence = sequences[i]
                    shuffled_sequences.append(
                        "".join(window) + sequence[len(window) * window_size :]
                    )

            if len(shuffled_sequences) > self.num_shuffles:
                break

        return shuffled_sequences[: self.num_shuffles]

    def _validate(
        self, wt_sequences: List[str], sequences: List[str]
    ) -> List[List[str]]:
        """
        Args:
            wt_sequences (List[str]): 野生型のタンパク質のリスト
            sequences (List[str]): タンパク質のリスト

        Returns:
            List[List[str]]: 変異されたタンパク質のリスト
        """
        outputs: List[List[str]] = []

        for wt_seq in wt_sequences:
            output = [wt_seq]
            for seq in sequences:
                if len(seq) != len(wt_seq):
                    continue
                diff = sum(aa != bb for aa, bb in zip(wt_seq, seq))
                if diff <= 4:
                    output.append(seq)
            outputs.append(output)

        return outputs

    def fold(self) -> None:
        records: List[SeqRecord] = []

        for id, seq in zip(self.ids, self.wt_sequences):
            record = SeqRecord(Seq(seq), id=id, description="")
            records.append(record)

        fasta_path = self.project_dir / "sampler" / "query.fasta"

        SeqIO.write(records, fasta_path, "fasta")

        subprocess.run(["colabfold_batch", fasta_path, self.structure_dir], check=True)

    def sample(self) -> Dict[str, List[str]]:
        """
        Returns:
            Dict[str, List[str]]: サンプリングされたタンパク質の辞書
        """
        samples: Dict[str, List[str]] = {}

        for id, wt_seq in zip(self.ids, self.wt_sequences):
            sequences = self._mutate(wt_seq)
            samples[id] = [wt_seq, *sequences]

        sequences: List[str] = self._shuffle(self.wt_sequences)
        results: List[List[str]] = self._validate(self.wt_sequences, sequences)

        for id, result in zip(self.ids, results):
            samples[id].extend(result[1:])

        for id, sample in samples.items():
            records: List[SeqRecord] = []
            for i, seq in enumerate(sample, start=1):
                record = SeqRecord(Seq(seq), id=str(i), description="")
                records.append(record)
            fasta_path = self.sequence_dir / f"{id}.fasta"
            SeqIO.write(records, fasta_path, "fasta")

        return samples

    def load(self) -> Dict[str, List[str]]:
        """
        Returns:
            Dict[str, List[str]]: サンプリングされたタンパク質の辞書
        """
        samples: Dict[str, List[str]] = {}

        for id in self.ids:
            fasta_path = self.sequence_dir / f"{id}.fasta"
            records = list(SeqIO.parse(fasta_path, "fasta"))
            samples[id] = [str(rec.seq) for rec in records]

        return samples
