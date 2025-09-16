import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import joblib
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
from tqdm import trange
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

    def __init__(self, cfg: Mapping[str, Any], name: str) -> None:
        """
        Args:
            cfg (Mapping[str, Any]): 設定
            name (str): 予測器の名前
        """
        cfg = SimpleNamespace(**cfg)

        self.seed: int = cfg.seed
        self.debug: bool = cfg.debug
        self.device: torch.device = cfg.device

        self.name: str = name

        self.project_dir: Path = Path("runs") / cfg.project

        self.weight_dir = self.project_dir / "predictor" / self.name / "weight"
        self.weight_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir: Path = (
            self.project_dir / "predictor" / self.name / "checkpoint"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size: int = cfg.batch_size
        self.test_size: float = cfg.test_size
        self.num_epochs: int = cfg.num_epochs
        self.num_trials: int = cfg.num_trials
        self.patience: int = cfg.patience

        self.mutate_per_samples: int = cfg.mutate_per_samples
        self.num_mutations: int = cfg.num_mutations
        self.destruct_per_samples: int = cfg.destruct_per_samples
        self.num_destructions: int = cfg.num_destructions
        self.noise_ratio: float = cfg.noise_ratio

        self.model_name_or_path: str = cfg.model_name_or_path
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        csv_path = self.project_dir / "data" / "input.csv"
        df = pd.read_csv(csv_path)
        self.sequences: List[str] = df["sequence"].tolist()
        self.labels: List[float] = df[self.name].tolist()

        random.seed(self.seed)
        np.random.seed(self.seed)

    def _prepare(self) -> Tuple[List[str], List[str], List[float], List[float]]:
        """
        Returns:
            Tuple[List[str], List[str], List[float], List[float]]: データセット
        """
        X = np.array(self.sequences)
        y = np.array(self.labels)

        threshold = np.quantile(y, 0.98)
        X = X[y <= threshold]
        y = y[y <= threshold]

        bins = int(np.sqrt(len(y)))
        stratify = pd.qcut(y, bins, labels=False)

        X_train, X_test, y_train, y_test = train_test_split(
            X.tolist(),
            y.tolist(),
            test_size=self.test_size,
            random_state=self.seed,
            stratify=stratify,
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

        def destruct(sequence: str, positions: Optional[List[int]] = None) -> str:
            """
            Args:
                sequence (str): 破壊するタンパク質
                positions (Optional[List[int]]): 破壊する位置

            Returns:
                str: 破壊されたタンパク質
            """
            sequence = list(sequence)

            if positions is None:
                positions = random.sample(range(len(sequence)), k=self.num_destructions)

            for pos in positions:
                aa = sequence[pos]
                candidates = conserve(aa, upper=-1)
                if not candidates:
                    candidates = conserve(aa, upper=0)
                if not candidates:
                    candidates = [bb for bb in blosum62.alphabet if bb != aa]
                sequence[pos] = random.choice(candidates)

            sequence = "".join(sequence)

            return sequence

        def mutate(sequence: str, positions: Optional[List[int]] = None) -> str:
            """
            Args:
                sequence (str): 変異させるタンパク質
                positions (Optional[List[int]]): 変異させる位置

            Returns:
                str: 変異されたタンパク質
            """
            sequence = list(sequence)

            if positions is None:
                positions = random.sample(range(len(sequence)), k=self.num_mutations)

            for pos in positions:
                aa = sequence[pos]
                candidates = conserve(aa, lower=1)
                if not candidates:
                    candidates = conserve(aa, lower=0)
                if not candidates:
                    candidates = [bb for bb in blosum62.alphabet if bb != aa]
                sequence[pos] = random.choice(candidates)

            sequence = "".join(sequence)

            return sequence

        results = []

        for sequence in X_train:
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
                    logits = model(masked_input_ids).logits

                logits = logits[0, i]  # (V,)
                probs = torch.log_softmax(logits, dim=-1)
                score = probs[input_ids[0, i]].item()

                scores.append(score)

            results.append((sequence, scores))

        weights = np.array([np.mean(res[1]) for res in results])
        weights = np.exp(weights - weights.max())
        results = random.choices(results, weights=weights, k=self.destruct_per_samples)

        X: List[str] = []
        y: List[float] = []

        for sequence, scores in results:
            top_k = np.argsort(scores)[-self.num_destructions * 3 :]
            probs = np.exp(np.array(scores)[top_k])
            probs = probs / probs.sum()
            indices = np.random.choice(
                top_k, size=self.num_destructions, replace=False, p=probs
            )
            sequence = destruct(sequence, indices.tolist())
            X.append(sequence)
            y.append(1e-6)

        X_train.extend(X)
        y_train.extend(y)

        results = list(zip(X_train, y_train))
        results = random.choices(results, k=self.mutate_per_samples)

        X: List[str] = []
        y: List[float] = []

        for sequence, label in results:
            sequence = mutate(sequence)
            noise = random.uniform(-self.noise_ratio, self.noise_ratio)
            X.append(sequence)
            y.append(label * (1 + noise))

        X_train.extend(X)
        y_train.extend(y)

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        method = "box-cox" if np.min(y_train) > 0 else "yeo-johnson"
        self.transformer = PowerTransformer(method=method)
        y_train = self.transformer.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = self.transformer.transform(y_test.reshape(-1, 1)).flatten()
        joblib.dump(self.transformer, self.weight_dir / "transformer.joblib")

        self.scaler = RobustScaler()
        y_train = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = self.scaler.transform(y_test.reshape(-1, 1)).flatten()
        joblib.dump(self.scaler, self.weight_dir / "scaler.joblib")

        y_train = y_train.tolist()
        y_test = y_test.tolist()

        return X_train, X_test, y_train, y_test

    def train(self) -> None:
        X_train, X_test, y_train, y_test = self._prepare()

        train_dataset = Dataset.from_dict(
            {"sequence": X_train, "labels": y_train}
        ).shuffle(seed=self.seed)
        eval_dataset = Dataset.from_dict(
            {"sequence": X_test, "labels": y_test}
        ).shuffle(seed=self.seed)

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

        best_params: Dict[str, float] = {}

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
                lora_dropout = best_params.get("lora_dropout", 0.0)
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
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=self.num_epochs,
            save_strategy="no",
            save_total_limit=2,
            seed=self.seed,
            fp16=True,
            fp16_full_eval=True,
            metric_for_best_model="eval_r2",
            greater_is_better=True,
            report_to="none",
        )

        data_collator = DataCollatorWithPadding(self.tokenizer)

        def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
            """
            Args:
                pred (EvalPrediction): 予測オブジェクト

            Returns:
                Dict[str, float]: 評価指標
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

        def compute_objective(metrics: Dict[str, float]) -> float:
            """
            Args:
                metrics (Dict[str, float]): 評価指標

            Returns:
                float: 目的関数の値
            """
            return metrics["eval_r2"]

        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hp_space,
            n_trials=self.num_trials,
            compute_objective=compute_objective,
        )

        best_params = {
            "r": best_trial.hyperparameters["r"],
            "lora_alpha": best_trial.hyperparameters["lora_alpha"],
            "lora_dropout": best_trial.hyperparameters["lora_dropout"],
        }

        args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            eval_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=best_trial.hyperparameters["learning_rate"],
            weight_decay=best_trial.hyperparameters["weight_decay"],
            num_train_epochs=self.num_epochs,
            warmup_ratio=best_trial.hyperparameters["warmup_ratio"],
            save_strategy="epoch",
            save_total_limit=2,
            seed=self.seed,
            fp16=True,
            fp16_full_eval=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_r2",
            greater_is_better=True,
            report_to="none",
        )

        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)],
        )

        trainer.train()

        self.model = trainer.model.eval()
        torch.save(self.model, self.weight_dir / "model.pt")

    def load(self) -> None:
        self.model = torch.load(self.weight_dir / "model.pt", weights_only=False)
        self.model.to(self.device, dtype=torch.float16).eval()
        self.transformer = joblib.load(self.weight_dir / "transformer.joblib")
        self.scaler = joblib.load(self.weight_dir / "scaler.joblib")

    def predict(self, sequences: List[str]) -> List[float]:
        """
        Args:
            sequences (List[str]): タンパク質のリスト

        Returns:
            List[float]: 予測値のリスト
        """
        outputs: List[float] = []

        for i in trange(
            0,
            len(sequences),
            self.batch_size,
            desc="Predicting fitness",
            disable=not self.debug,
        ):
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

        outputs = torch.cat(outputs, dim=0).cpu().numpy()
        outputs = self.scaler.inverse_transform(outputs.reshape(-1, 1))
        outputs = self.transformer.inverse_transform(outputs).flatten()
        outputs = outputs.tolist()

        return outputs
