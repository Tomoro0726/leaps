import gc
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

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
        self.patience: int = cfg.patience

        self.mutate_per_samples: int = cfg.mutate_per_samples
        self.num_mutations: int = cfg.num_mutations

        self.destruct_per_samples: int = cfg.destruct_per_samples
        self.num_destructions: int = cfg.num_destructions

        self.noise_ratio: float = cfg.noise_ratio
        self.num_trials: int = cfg.num_trials

        self.model_name_or_path: str = cfg.model_name_or_path
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        csv_path = self.project_dir / "data" / "input.csv"
        df = pd.read_csv(csv_path)
        self.sequences: List[str] = df["sequence"].tolist()
        self.labels: List[float] = df[self.name].tolist()

        random.seed(self.seed)
        np.random.seed(self.seed)

    def _prepare(self) -> None:
        X = np.array(self.sequences)
        y = np.array(self.labels)

        threshold = np.quantile(y, 0.98)
        mask = y <= threshold

        X = X[mask]
        y = y[mask]

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            X.tolist(),
            y.tolist(),
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
                aa = sequence[pos]
                candidates = conserve(aa, upper=-1)
                if not candidates:
                    candidates = conserve(aa, upper=0)
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
                aa = sequence[pos]
                candidates = conserve(aa, lower=1)
                if not candidates:
                    candidates = conserve(aa, lower=0)
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
            y_train.append(1e-6)

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

        method = "box-cox" if np.min(self.y_train) > 0 else "yeo-johnson"
        self.transformer = PowerTransformer(method=method)
        self.y_train = self.transformer.fit_transform(
            np.array(self.y_train).reshape(-1, 1)
        ).flatten()
        self.y_test = self.transformer.transform(
            np.array(self.y_test).reshape(-1, 1)
        ).flatten()
        joblib.dump(self.transformer, self.weight_dir / "transformer.joblib")

        self.scaler = RobustScaler()
        self.y_train = self.scaler.fit_transform(
            np.array(self.y_train).reshape(-1, 1)
        ).flatten()
        self.y_test = self.scaler.transform(
            np.array(self.y_test).reshape(-1, 1)
        ).flatten()
        joblib.dump(self.scaler, self.weight_dir / "scaler.joblib")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    def train(self) -> None:
        self._prepare()

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
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=self.num_epochs,
            seed=self.seed,
            report_to="none",
            save_strategy="no",
            save_total_limit=2,
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

        outputs = torch.cat(outputs, dim=0).to(torch.float32).cpu().numpy()
        outputs = self.scaler.inverse_transform(outputs.reshape(-1, 1))
        outputs = self.transformer.inverse_transform(outputs).flatten()
        outputs = outputs.tolist()

        gc.collect()
        torch.cuda.empty_cache()

        return outputs
