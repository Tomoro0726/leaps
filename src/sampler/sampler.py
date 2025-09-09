import gc
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List, Mapping

import pandas as pd
import torch
from Bio import SeqIO
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein, to_pdb


class Sampler:
    """
    タンパク質をサンプリングするクラス
    """

    def __init__(self, cfg: Mapping[str, object]) -> None:
        """
        Args:
            cfg (Mapping[str, object]): 設定
        """
        self.cfg: SimpleNamespace = SimpleNamespace(**cfg)

        self.seed: int = self.cfg.seed
        self.device: torch.device = self.cfg.device

        self.project_dir: Path = Path("runs") / self.cfg.project

        self.sequence_dir: Path = self.project_dir / "sampler" / "sequence"
        self.sequence_dir.mkdir(parents=True, exist_ok=True)

        self.structure_dir: Path = self.project_dir / "sampler" / "structure"
        self.structure_dir.mkdir(parents=True, exist_ok=True)

        csv_path: Path = self.project_dir / "data" / "input.csv"
        df = pd.read_csv(csv_path)
        self.sequences: List[str] = df["sequence"].tolist()

        self.num_shuffles: int = self.cfg.num_shuffles
        self.window_sizes: List[int] = self.cfg.window_size
        self.shuffle_rate: float = self.cfg.shuffle_rate

        random.seed(self.seed)

        self.model_name_or_path: str = self.cfg.model_name_or_path
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

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
        shuffled_sequence: List[str] = []

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
                    shuffled_sequence.append(
                        "".join(window) + sequence[len(window) * window_size :]
                    )

            if len(shuffled_sequence) > self.num_shuffles:
                break

        return shuffled_sequence[: self.num_shuffles]

    def _validate(self, sequences: List[str]) -> List[List[str]]:
        """
        Args:
            sequences (List[str]): タンパク質のリスト

        Returns:
            List[List[str]]: 変異されたタンパク質のリスト
        """
        outputs: List[List[str]] = []

        for wt_seq in self.sequences:
            col = [wt_seq]
            for seq in sequences:
                if len(seq) != len(wt_seq):
                    continue
                diff = sum(aa != bb for aa, bb in zip(wt_seq, seq))
                if diff <= 4:
                    col.append(seq)
            outputs.append(col)

        return outputs

    @torch.inference_mode()
    def _get_pdbstr(self, sequence: str) -> str:
        """
        Args:
            sequence (str): タンパク質

        Returns:
            str: PDB形式の文字列
        """
        if self.model is None:
            self.model = EsmForProteinFolding.from_pretrained(
                self.model_name_or_path,
                device_map="auto",
            ).eval()

        inputs = self.tokenizer(
            sequence,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        atom_positions = atom14_to_atom37(outputs.positions[-1], outputs)

        protein = OFProtein(
            aatype=outputs.aatype[0].cpu().numpy(),
            atom_positions=atom_positions[0].cpu().numpy(),
            atom_mask=outputs.atom37_atom_exists[0].cpu().numpy(),
            residue_index=outputs.residue_index[0].cpu().numpy() + 1,
            b_factors=outputs.plddt[0].cpu().numpy(),
        )

        pdbstr = to_pdb(protein)

        return pdbstr

    def sample(self) -> List[List[str]]:
        """
        Args:
            sequences (List[str]): タンパク質のリスト

        Returns:
            List[List[str]]: サンプルのリスト
        """
        for i, seq in enumerate(self.sequences, start=1):
            pdbstr = self._get_pdbstr(seq)
            save_path = self.structure_dir / f"{i}.pdb"
            with save_path.open("w") as f:
                f.write(pdbstr)

        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

        outputs: List[List[str]] = []

        for wt_seq in self.sequences:
            col: List[str] = [wt_seq]
            for seq in self._mutate(wt_seq):
                col.append(seq)
            outputs.append(col)

        sequences: List[str] = self._shuffle(self.sequences)
        results: List[List[str]] = self._validate(sequences)

        for i in range(len(outputs)):
            for seq in results[i][1:]:
                outputs[i].append(seq)

        for i, col in enumerate(outputs, start=1):
            records: List[SeqRecord] = [
                SeqRecord(Seq(seq), id=str(j), description="")
                for j, seq in enumerate(col)
            ]
            save_path = self.sequence_dir / f"{i}.fasta"
            SeqIO.write(records, save_path, "fasta")

        return outputs

    def load(self) -> List[List[str]]:
        """
        Returns:
            List[List[str]]: タンパク質のリスト
        """
        outputs: List[List[str]] = []
        fasta_paths = sorted(
            self.sequence_dir.glob("*.fasta"), key=lambda p: int(p.stem)
        )
        for fasta_path in fasta_paths:
            records = list(SeqIO.parse(fasta_path, "fasta"))
            col = [str(rec.seq) for rec in records]
            outputs.append(col)

        return outputs
