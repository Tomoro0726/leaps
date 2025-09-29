from pathlib import Path
import shutil
from typing import Dict, List

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd

from src.config import Config
from src.early_stopper import EarlyStopper
from src.evaluator.fitness import Fitness
from src.evaluator.hamiltonian import Hamiltonian
from src.evaluator.likelihood import Likelihood
from src.generator import Generator
from src.predictor import Predictor
from src.sampler import Sampler
from src.state import State


class Runner:
    """
    実行を管理するクラス
    """

    def __init__(self, cfg: Config, csv_path: str | Path = "data/input.csv") -> None:
        """
        Args:
            cfg (Config): 設定
            csv_path (str | Path): CSVファイルのパス
        """
        self.state = State()

        self.project_dir: Path = Path("runs") / cfg.project

        data_dir: Path = self.project_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        src = csv_path
        dst: Path = data_dir / "input.csv"
        shutil.copy2(src, dst)

        self.input_dir: Path = self.project_dir / "runner" / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir: Path = self.project_dir / "runner" / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_iterations: int = cfg["runner"]["num_iterations"]
        self.num_sequences: int = cfg["runner"]["num_sequences"]

        self.sampler = Sampler(cfg["sampler"])

        self.predictors: Dict[str, Predictor] = {}
        for k, v in cfg["predictor"].items():
            if isinstance(v, dict):
                self.predictors[k] = Predictor(v, k)

        self.hamiltonian = Hamiltonian(cfg["evaluator"]["hamiltonian"], self.state)
        self.likelihood = Likelihood(cfg["evaluator"]["likelihood"], self.state)

        self.fitnesses: Dict[str, Fitness] = {}
        for k, v in cfg["evaluator"].items():
            if k in ["hamiltonian", "likelihood"]:
                continue

            if isinstance(v, dict):
                self.fitnesses[k] = Fitness(v, self.state, k, self.predictors[k])

        self.generator = Generator(cfg["generator"], self.state)

        self.early_stopper = EarlyStopper(cfg["early_stopper"], self.state)

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
            iteration (int): イテレーション数

        Returns:
            List[str]: スクリーニングされたタンパク質のリスト
        """
        outputs: List[str] = []

        for fitness in self.fitnesses.values():
            outputs.extend(fitness.filter(sequences, strategy="parallel"))

        for fitness in self.fitnesses.values():
            sequences = fitness.filter(sequences, strategy="series")
        outputs.extend(sequences)

        outputs = self._unique(outputs)

        return outputs

    def _save(self, sequences: List[str], save_path: str | Path) -> None:
        """
        Args:
            sequences (List[str]): 保存したいタンパク質のリスト
            save_path (str | Path): 保存先のパス
        """
        records: List[SeqRecord] = []

        for i, seq in enumerate(sequences, start=1):
            record = SeqRecord(seq=Seq(seq), id=str(i), description="")
            records.append(record)

        SeqIO.write(records, save_path, "fasta")

    def run(self) -> List[str]:
        """
        Returns:
            List[str]: 最終的なタンパク質のリスト
        """
        self.sampler.fold()
        samples: Dict[str, List[str]] = self.sampler.sample()

        for predictor in self.predictors.values():
            try:
                predictor.load()
            except Exception:
                predictor.train()

        model_path = (
            self.project_dir / "evaluator" / "hamiltonian" / "weight" / "model.params"
        )
        if not model_path.exists():
            self.hamiltonian.train()

        prev: List[str] | None = None
        next: List[str] | None = None

        for iteration in range(1, self.num_iterations + 1):
            self.state.iteration = iteration

            sequences: List[str] = []

            fasta_path = self.input_dir / f"iter{iteration}.fasta"

            if fasta_path.exists():
                records = list(SeqIO.parse(fasta_path, "fasta"))
                sequences = [str(rec.seq) for rec in records]
            else:
                if iteration == 1:
                    sequences = self.likelihood.filter(samples)
                    sequences = self._unique(sequences)
                    sequences = self.hamiltonian.filter(sequences)
                    sequences = self._pipeline(sequences)
                else:
                    sequences = self.hamiltonian.filter(next)
                    sequences = self._pipeline(sequences)
                self._save(sequences, fasta_path)

            try:
                self.generator.load()
            except Exception:
                self.generator.train(sequences)

            fasta_path = self.output_dir / f"iter{iteration}.fasta"
            if fasta_path.exists():
                records = list(SeqIO.parse(fasta_path, "fasta"))
                sequences = [str(rec.seq) for rec in records]
            else:
                sequences = self.generator.generate(
                    100000 if iteration == 1 else self.num_sequences
                )
                sequences = self._unique(sequences)
                self._save(sequences, fasta_path)

            next = sequences

            if prev is not None and self.early_stopper(prev, next):
                break

            prev = next

        sequences = next

        data = {}
        results: Dict[str, pd.Series] = {}

        for name, fitness in self.fitnesses.items():
            scores = fitness.score(sequences)
            results[name] = pd.Series(scores, index=sequences)
            ranks = {
                seq: j + 1 for j, (seq, _) in enumerate(fitness.sort(sequences, scores))
            }
            data[name] = pd.Series(ranks)

        df = pd.DataFrame(data, index=sequences)

        k = 60
        score = (1.0 / (k + df)).sum(axis=1)
        sequences = score.sort_values(ascending=False).index.tolist()

        fasta_path = self.output_dir / "result.fasta"
        self._save(sequences[:10], fasta_path)

        csv_path = self.output_dir / "result.csv"
        names = list(self.fitnesses.keys())

        df = pd.DataFrame(
            {name: results[name].loc[sequences[:10]].values for name in names},
            index=sequences[:10],
        )
        df = df.rename_axis("sequence").reset_index()
        df.insert(0, "id", range(1, len(df) + 1))
        df.to_csv(csv_path, index=False)

        return sequences
