from pathlib import Path
from types import SimpleNamespace
from typing import List, Mapping, Optional

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src.predictor.predictor import Predictor
from src.evaluator.fitness import Fitness
from src.evaluator.likelihood import Likelihood


class Evaluator:
    """
    タンパク質をスクリーニングするクラス
    """

    def __init__(
        self,
        cfg: Mapping[str, object],
        predictors: Optional[Mapping[str, Predictor]] = None,
    ) -> None:
        """
        Args:
            cfg (Mapping[str, object]): 設定
            predictors (Optional[Mapping[str, Predictor]): 予測器の辞書
        """
        cfg = SimpleNamespace(**cfg)

        self.project_dir: Path = Path("runs") / cfg.project

        self.save_dir: Path = self.project_dir / "evaluator"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.likelihood = Likelihood(cfg.likelihood)

        self.fitnesses: List[Fitness] = []
        for k, v in (predictors or {}).items():
            self.fitnesses.append(Fitness(getattr(cfg, k), v))

    def _unique(self, sequences: List[str]) -> List[str]:
        """
        Args:
            sequences (List[str]): タンパク質のリスト

        Returns:
            List[str]: 重複を除去したタンパク質のリスト
        """
        visited = set()
        outputs: List[str] = []
        for seq in sequences:
            if seq not in visited:
                visited.add(seq)
                outputs.append(seq)

        return outputs

    def _pipeline(self, sequences: List[str]) -> List[str]:
        """
        Args:
            sequences (List[str]): タンパク質のリスト

        Returns:
            List[str]: スクリーニングされたタンパク質のリスト
        """
        outputs = []

        for fitness in self.fitnesses:
            outputs.extend(fitness.filter(sequences, strategy="parallel"))

        outputs = self._unique(outputs)

        for fitness in self.fitnesses:
            sequences = fitness.filter(sequences, strategy="series")

        outputs.extend(sequences)
        outputs = self._unique(outputs)

        return outputs

    def filter(self, iteration: int, inputs: List[List[str]] | List[str]) -> List[str]:
        """
        Args:
            iteration (int): イテレーション
            inputs (List[List[str]] | List[str]): スクリーニングするタンパク質のリスト

        Returns:
            List[str]: スクリーニングされたタンパク質のリスト
        """
        records: List[SeqRecord] = []

        save_path = self.save_dir / f"{iteration}.fasta"
        if save_path.exists():
            records = list(SeqIO.parse(save_path, "fasta"))
            outputs = [str(rec.seq) for rec in records]

            return outputs

        if iteration == 1:
            count = 0
            for i, sequences in enumerate(inputs, start=1):
                pdb_path = self.project_dir / "sampler" / "structure" / f"{i}.pdb"
                sequences = self.likelihood.filter(sequences, pdb_path)
                sequences = self._pipeline(sequences)

                for seq in sequences:
                    records.append(
                        SeqRecord(seq=Seq(seq), id=str(count), description="")
                    )
                    count += 1
        else:
            sequences: List[str] = []
            if isinstance(inputs[0], list):
                sequences = [seq for col in inputs for seq in col]
            sequences = self._pipeline(sequences)

            for i, seq in enumerate(sequences, start=1):
                records.append(SeqRecord(seq=Seq(seq), id=str(i), description=""))

        SeqIO.write(records, save_path, "fasta")

        outputs = [str(rec.seq) for rec in records]

        return outputs
