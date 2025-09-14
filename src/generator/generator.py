from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

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

from src.state.state import State

logging.set_verbosity_error()


class Generator:
    """
    タンパク質を生成するクラス
    """

    def __init__(self, cfg: Mapping[str, Any], state: State) -> None:
        """
        Args:
            cfg (Mapping[str, Any]): 設定
        """
        cfg = SimpleNamespace(**cfg)
        self.state = state

        self.seed: int = cfg.seed
        self.debug: bool = cfg.debug
        self.device: torch.device = cfg.device

        self.project_dir: Path = Path("runs") / cfg.project

        self.weight_dir = self.project_dir / "generator" / "weight"
        self.weight_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir: Path = self.project_dir / "generator" / "checkpoint"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size: int = cfg.batch_size
        self.test_size: float = cfg.test_size

        self.r: int = cfg.r
        self.lora_alpha: int = cfg.lora_alpha
        self.lora_dropout: float = cfg.lora_dropout

        self.num_epochs: int = cfg.num_epochs
        self.patience: int = cfg.patience

        self.max_new_token: int = cfg.max_new_token
        self.prompt: str = cfg.prompt

        self.model_name_or_path: str = cfg.model_name_or_path
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(
        self,
        sequences: List[str],
    ) -> None:
        """
        Args:
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
            save_total_limit=2,
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
        torch.save(self.model, self.weight_dir / f"iter{self.state.iteration}.pt")

    def load(self) -> None:
        self.model = torch.load(
            self.weight_dir / f"iter{self.state.iteration}.pt", weights_only=False
        )
        self.model.to(self.device, dtype=torch.float16).eval()

    @torch.inference_mode()
    def generate(
        self,
        num_sequences: Optional[int] = 1,
        top_k: Optional[int] = 10,
        top_p: Optional[float] = 0.90,
        temperature: Optional[float] = 1.5,
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

        def prefix_allowed_tokens_fn(_batch_id, _input_ids):
            return [
                self.tokenizer.convert_tokens_to_ids(aa)
                for aa in IUPACData.protein_letters
            ]

        sequences: List[str] = []

        for _ in trange(
            0,
            num_sequences,
            self.batch_size,
            desc="Generating sequences",
            disable=not self.debug,
        ):
            batch_size = min(self.batch_size, num_sequences - len(sequences))

            input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt").to(
                self.device
            )

            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_token,
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
