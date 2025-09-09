from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import optuna
import torch
from Bio.Data import IUPACData
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from tqdm import trange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    logging,
)

logging.set_verbosity_error()


class Generator:
    """
    タンパク質を生成するクラス
    """

    def __init__(self, cfg: Mapping[str, object] = None) -> None:
        """
        Args:
            cfg (Mapping[str, object]): 設定
        """
        self.cfg = SimpleNamespace(**cfg)

        self.seed: int = self.cfg.seed
        self.debug: bool = self.cfg.debug
        self.device: torch.device = self.cfg.device

        self.project_dir: Path = Path("runs") / self.cfg.project

        self.save_dir = self.project_dir / "generator" / "weight"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir: Path = self.project_dir / "generator" / "checkpoint"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size: int = self.cfg.batch_size
        self.test_size: float = self.cfg.test_size

        self.r: int = self.cfg.r
        self.lora_alpha: int = self.cfg.lora_alpha
        self.lora_dropout: float = self.cfg.lora_dropout

        self.num_epochs: int = self.cfg.num_epochs
        self.patience: int = self.cfg.patience

        self.num_samples: int = self.cfg.num_samples
        self.num_trials: int = self.cfg.num_trials
        self.prompt: str = self.cfg.prompt

        self.model_name_or_path: str = self.cfg.model_name_or_path
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.var: float = 0.0
        self.best_params: Dict[str, Any] = {}

    @torch.inference_mode()
    def _embed(self, sequences: List[str]) -> torch.Tensor:
        """
        Args:
            sequences (List[str]): タンパク質のリスト

        Returns:
            torch.Tensor: 埋め込みベクトル
        """
        results: List[torch.Tensor] = []

        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            attention_mask = inputs["attention_mask"]
            extended_attention_mask = attention_mask.unsqueeze(-1)
            pooled_output = (last_hidden_state * extended_attention_mask).sum(
                1
            ) / extended_attention_mask.sum(1)
            results.append(pooled_output.detach().cpu())

        results = torch.cat(results, dim=0)

        return results

    def _objective(self, trial: optuna.trial.Trial) -> float:
        temperature = trial.suggest_float("temperature", 0.01, 1.5)
        sequences = self.generate(self.num_samples, temperature=temperature)

        try:
            outputs = self._embed(sequences).numpy()
            var = float(np.var(outputs, axis=0).mean())
            trial.set_user_attr("var", var)

            if self.var <= 0:
                return 0.0

            lower = 0.60 * self.var
            upper = 0.90 * self.var
            penalty = max(0.0, lower - var) + max(0.0, var - upper)
            score = 1.0 / (1.0 + penalty)

            return score

        except Exception:
            return 0.0

    def tune(self) -> None:
        if not self.debug:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        if not self.best_params:
            sequences = self.generate(self.num_samples, temperature=1.5)
            outputs = self._embed(sequences).numpy()
            var = float(np.var(outputs, axis=0).mean())
            self.var = var
            self.best_params = {"temperature": 1.5}

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )
        study.optimize(self._objective, n_trials=self.num_trials)

        self.var = study.best_trial.user_attrs["var"]
        self.best_params = study.best_params

    def train(self, iteration: int, sequences: List[str]) -> None:
        """
        Args:
            iteration (int): イテレーション数
            sequences (List[str]): タンパク質のリスト
        """
        train, test = train_test_split(
            sequences,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True,
        )

        train_dataset = Dataset.from_dict({"sequence": train})
        test_dataset = Dataset.from_dict({"sequence": test})

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
            tokenize,
            batched=True,
            remove_columns=["sequence"],
        )
        eval_dataset = test_dataset.map(
            tokenize, batched=True, remove_columns=["sequence"]
        )

        def model_init() -> PeftModel:
            """
            Returns:
                PeftModel: PEFTモデル
            """
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device)
            model.config.num_hidden_layers = len(model.transformer.h)

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["qkv_proj", "out_proj"],
            )
            model = get_peft_model(model, lora_config)

            model.print_trainable_parameters()

            return model

        args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            eval_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=self.num_epochs,
            seed=self.seed,
            report_to="none",
            save_strategy="epoch",
            save_total_limit=1,
            fp16=True,
            fp16_full_eval=True,
            load_best_model_at_end=True,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)],
        )

        trainer.train()

        self.model = trainer.model.eval()
        self.model.save_pretrained(self.save_dir / str(iteration))
        self.tokenizer.save_pretrained(self.save_dir / str(iteration))

    def load(self, iteration: int) -> None:
        """
        Args:
            iteration (int): イテレーション数
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(self.device)

        self.model = (
            PeftModel.from_pretrained(
                model,
                self.save_dir / str(iteration),
                is_trainable=False,
            )
            .to(self.device)
            .eval()
        )
        self.model.config.num_hidden_layers = len(self.model.transformer.h)

    @torch.inference_mode()
    def generate(
        self,
        num_sequences: Optional[int] = 1,
        top_k: Optional[int] = 10,
        top_p: Optional[float] = 0.90,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        Args:
            num_sequences (Optional[int]): 生成するタンパク質の数
            top_k (Optional[int]): Top-K
            top_p (Optional[float]): Top-P
            temperature (Optional[float]): 温度

        Returns:
            List[str]: 生成されたタンパク質のリスト
        """
        if temperature is None:
            temperature = self.best_params.get("temperature", 1.0)

        def prefix_allowed_tokens_fn(_batch_id, _input_ids):
            return [
                self.tokenizer.convert_tokens_to_ids(aa)
                for aa in IUPACData.protein_letters
            ]

        sequences: List[str] = []

        for _ in trange(0, num_sequences, self.batch_size, disable=not self.debug):
            batch_size = min(self.batch_size, num_sequences - len(sequences))

            input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt").to(
                self.device
            )

            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                no_repeat_ngram_size=5,
                num_return_sequences=batch_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            sequences.extend(outputs)

        return sequences
