import gc
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import optuna
import pandas as pd
import torch
from Bio.Align import substitution_matrices
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, RobustScaler
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EsmForMaskedLM,
    EsmModel,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    logging,
)

from src.predictor.regressor import Regressor

logging.set_verbosity_error()


class Predictor:
    """
    タンパク質の機能を予測するクラス
    """

    def __init__(self, cfg: Mapping[str, object]) -> None:
        """
        Args:
            cfg (Mapping[str, object]): 設定
        """
        self.cfg = SimpleNamespace(**cfg)

        self.seed: int = self.cfg.seed
        self.debug: bool = self.cfg.debug
        self.device: torch.device = self.cfg.device

        project_dir: Path = Path("runs") / self.cfg.project

        self.csv_path: Path = project_dir / "data" / "input.csv"

        self.save_dir = project_dir / "predictor" / self.cfg.target / "weight"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir: Path = (
            project_dir / "predictor" / self.cfg.target / "checkpoint"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size: int = self.cfg.batch_size

        self.target: str = self.cfg.target
        self.test_size: float = self.cfg.test_size

        self.num_epochs: int = self.cfg.num_epochs
        self.patience: int = self.cfg.patience

        self.mutate_per_samples: int = self.cfg.mutate_per_samples
        self.num_mutations: int = self.cfg.num_mutations

        self.destruct_per_samples: int = self.cfg.destruct_per_samples
        self.num_destructions: int = self.cfg.num_destructions

        self.noise_ratio: float = self.cfg.noise_ratio
        self.num_trials: int = self.cfg.num_trials

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.model_name_or_path: str = self.cfg.model_name_or_path
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        self._prepare()

    def _prepare(self) -> None:
        df = pd.read_csv(self.csv_path)

        threshold = df[self.target].quantile(0.98)
        df = df[df[self.target] <= threshold]

        method = "box-cox" if (df[self.target] > 0).all() else "yeo-johnson"
        self.transformer = PowerTransformer(method=method)
        df[self.target] = self.transformer.fit_transform(
            df[self.target].to_numpy().reshape(-1, 1)
        ).flatten()

        sequences = df["sequence"].tolist()
        labels = df[self.target].tolist()

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            sequences,
            labels,
            test_size=self.test_size,
            random_state=self.seed,
        )

        model = (
            EsmForMaskedLM.from_pretrained(
                self.model_name_or_path, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )

        blosum62 = substitution_matrices.load("BLOSUM62")

        def conserve(
            aa: str,
            lower: int | None = None,
            upper: int | None = None,
        ) -> List[str]:
            """
            Args:
                aa (str): アミノ酸
                lower (int | None): 最小のスコア
                upper (int | None): 最大のスコア

            Returns:
                List[str]: 保存するアミノ酸のリスト
            """
            i = blosum62.alphabet.index(aa)

            return [
                bb
                for j, bb in enumerate(blosum62.alphabet)
                if aa != bb
                and (lower is None or blosum62[i, j] >= lower)
                and (upper is None or blosum62[i, j] <= upper)
            ]

        def destruct(sequence: str, positions: List[int] | None = None) -> str:
            """
            Args:
                sequence (str): 破壊するタンパク質
                positions (List[int] | None): 破壊する位置

            Returns:
                str: 破壊されたタンパク質
            """
            sequence = list(sequence)

            if positions is None:
                positions = random.sample(range(len(sequence)), k=self.num_destructions)

            for pos in positions:
                candidates = conserve(sequence[pos], upper=-1)
                sequence[pos] = random.choice(candidates)

            sequence = "".join(sequence)

            return sequence

        def mutate(sequence: str, positions: List[int] | None = None) -> str:
            """
            Args:
                sequence (str): 変異させるタンパク質
                positions (List[int] | None): 変異させる位置

            Returns:
                str: 変異されたタンパク質
            """
            sequence = list(sequence)

            if positions is None:
                positions = random.sample(range(len(sequence)), k=self.num_mutations)

            for pos in positions:
                candidates = conserve(sequence[pos], lower=1)
                sequence[pos] = random.choice(candidates)

            sequence = "".join(sequence)

            return sequence

        results = []

        for sequence in self.X_train[:]:
            input_ids = self.tokenizer.encode(
                sequence,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            scores = []
            for i in range(1, input_ids.shape[1] - 1):
                masked_input_ids = input_ids.clone()
                masked_input_ids[0, i] = self.tokenizer.mask_token_id

                with torch.no_grad():
                    logits = model(masked_input_ids.to(self.device)).logits

                logits = logits[0, i]  # (V,)
                probs = torch.log_softmax(logits, dim=-1)
                score = probs[input_ids[0, i]].item()

                scores.append(score)

            results.append((sequence, scores))

        results.sort(key=lambda res: np.mean(res[1]), reverse=True)

        X_train = []
        y_train = []

        for sequence, scores in results[: self.destruct_per_samples]:
            top_k = np.argsort(scores)[-self.num_destructions * 3 :]
            probs = np.exp(np.array(scores)[top_k])
            probs = probs / probs.sum()
            indices = np.random.choice(
                top_k, size=self.num_destructions, replace=False, p=probs
            )
            sequence = destruct(sequence, indices.tolist())
            X_train.append(sequence)
            y_train.append(0.0)

        self.X_train.extend(X_train)
        self.y_train.extend(y_train)

        results = list(zip(self.X_train, self.y_train))
        results = random.sample(results, k=self.mutate_per_samples)

        X_train = []
        y_train = []

        for sequence, label in results:
            sequence = mutate(sequence)
            noise = random.uniform(-self.noise_ratio, self.noise_ratio)
            X_train.append(sequence)
            y_train.append(label * (1 + noise))

        self.X_train.extend(X_train)
        self.y_train.extend(y_train)

        self.scaler = RobustScaler()
        self.y_train = self.scaler.fit_transform(
            np.array(self.y_train).reshape(-1, 1)
        ).flatten()
        self.y_test = self.scaler.transform(
            np.array(self.y_test).reshape(-1, 1)
        ).flatten()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    def train(self) -> None:
        if not self.debug:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        train_dataset = Dataset.from_dict(
            {"sequence": self.X_train, "labels": self.y_train}
        ).shuffle(seed=self.seed)
        eval_dataset = Dataset.from_dict(
            {"sequence": self.X_test, "labels": self.y_test}
        )

        def tokenize(examples: Dict[str, List[str]]) -> BatchEncoding:
            """
            Args:
                examples (Dict[str, List[str]]): バッチ

            Returns:
                BatchEncoding: エンコードされたバッチ
            """
            return self.tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=256,
            )

        train_dataset = train_dataset.map(
            tokenize, batched=True, remove_columns=["sequence"]
        )
        eval_dataset = eval_dataset.map(
            tokenize, batched=True, remove_columns=["sequence"]
        )

        best_params: Dict[str, Any] = {}

        def model_init(trial: Optional[optuna.Trial] = None) -> PeftModel:
            """
            Args:
                trial (Optional[optuna.Trial]): トライアルオブジェクト

            Returns:
                PeftModel: PEFTモデル
            """
            model = EsmModel.from_pretrained(
                self.model_name_or_path, torch_dtype=torch.float16
            )
            regressor = Regressor(model, model.config.hidden_size)

            if trial is None:
                r = best_params.get("r", 8)
                lora_alpha = best_params.get("lora_alpha", 16)
                lora_dropout = best_params.get("lora_dropout", 0.1)
            else:
                r = trial.suggest_categorical("r", [4, 8, 16, 32])
                lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32, 64])
                lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3, step=0.05)

            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "key", "value"],
            )
            model = get_peft_model(regressor, peft_config)
            model.print_trainable_parameters()

            return model

        args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            eval_strategy="epoch",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=self.num_epochs,
            seed=self.seed,
            report_to="none",
            save_strategy="no",
            save_total_limit=1,
            fp16=True,
            fp16_full_eval=True,
            metric_for_best_model="r2",
        )

        data_collator = DataCollatorWithPadding(self.tokenizer)

        def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
            """
            Args:
                pred (EvalPrediction): 予測オブジェクト

            Returns:
                Dict[str, float]: 決定係数
            """
            y_pred = pred.predictions.reshape(-1, 1)
            y_pred = self.transformer.inverse_transform(
                self.scaler.inverse_transform(y_pred)
            )

            y_true = pred.label_ids.reshape(-1, 1)
            y_true = self.transformer.inverse_transform(
                self.scaler.inverse_transform(y_true)
            )

            return {"r2": r2_score(y_true, y_pred)}

        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)],
        )

        def hp_space(trial: optuna.Trial) -> Dict[str, float]:
            """
            Args:
                trial (optuna.Trial): トライアルオブジェクト

            Returns:
                Dict[str, float]: ハイパーパラメータ
            """
            return {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 5e-6, 5e-4, log=True
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 1e-6, 1e-3, log=True
                ),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.10, 0.25),
            }

        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hp_space,
            n_trials=self.num_trials,
        )

        best_params = best_trial.hyperparameters

        trainer.args.learning_rate = best_params["learning_rate"]
        trainer.args.weight_decay = best_params["weight_decay"]
        trainer.args.warmup_ratio = best_params["warmup_ratio"]

        trainer.args.load_best_model_at_end = True
        trainer.args.save_strategy = "epoch"

        trainer = Trainer(
            model_init=model_init,
            args=trainer.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)],
        )

        trainer.train()

        self.model = trainer.model.eval()
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)

    def load(self) -> None:
        model = EsmModel.from_pretrained(
            self.model_name_or_path, torch_dtype=torch.float16
        ).to(self.device)
        regressor = Regressor(model, model.config.hidden_size)

        self.model = (
            PeftModel.from_pretrained(
                regressor,
                self.save_dir,
                is_trainable=False,
            )
            .to(self.device)
            .eval()
        )

    def predict(self, sequences: List[str]) -> List[float]:
        """
        Args:
            sequences (List[str]): タンパク質のリスト

        Returns:
            List[float]: 予測値のリスト
        """
        outputs: List[float] = []

        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            outputs.append(logits.detach().reshape(-1))

        outputs = torch.cat(outputs, dim=0).to(torch.float32).cpu().numpy()
        outputs = self.scaler.inverse_transform(outputs.reshape(-1, 1))
        outputs = self.transformer.inverse_transform(outputs).flatten()
        outputs = outputs.tolist()

        gc.collect()
        torch.cuda.empty_cache()

        return outputs
