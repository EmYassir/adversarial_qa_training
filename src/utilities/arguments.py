import dataclasses
import json
import os
from random import choices
import torch

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from transformers import SchedulerType

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_path: Optional[str] = field(
        default=None,
        metadata={"help":"Path to the training configuration with gernator, rank_discriminator, and answerability_discriminator configurations."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    num_dis_rounds: int = field(
        default=8,
        metadata={"help": "The number of discriminator rounds"},
    )
    num_gen_rounds: int = field(
        default=6,
        metadata={"help": "The number of generator rounds"},
    )
    answerability_heuristic: bool = field(
        default=False,
        metadata={"help": "Use heuristic for answerability instead of models. Utilizes BCELossWithLogits to compute answerability reward."},
    )
    ans_discriminator_weight: float = field(
        default=0.25,
        metadata={"help": "Weight to assign to answerability discriminator's reward for generator loss."},
    )

    regularizer_weight: float = field(
        default=1.,
        metadata={"help": "Weight to assign to regularizing factor in the reward for generator loss."},
    )

    hits_list: List[int] = field(
        default_factory=lambda: ([1, 3, 5, 10, 20, 30, 50]),
        metadata={"help": "List of N for Hits@N evaluation."},
    )

    device_map: List[int] = field(
        default_factory=lambda: ([0, 1, 2]),
        metadata={"help": "Device map for parallel/non parallel training."},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_path is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_path or --model_name_or_path"
            )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to use after saving to disk with a DatasetProcessor."}
    )
    is_tokenized: bool = field(
        default=False, metadata={"help": "If the dataset is already tokenized, loads it directly."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=350,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )



def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())



@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=2., metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(default=-1, metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    optimizer: str = field(default="SGD", metadata={"help": ("Optimizer. Choices: ['SGD' and 'AdamW'].")})

    logging_dir: Optional[str] = field(default_factory=default_logdir, metadata={"help": "Tensorboard log dir."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=100, metadata={"help": "Run an evaluation every X steps."})
    """ NOT SUPPORTED YET
    save_total_limit: Optional[int] = field(default=None, metadata={"help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    """
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})
    n_samples: int = field(default=50, metadata={"help": "Number of elements to sample from the generator scores"})
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    num_processes: int = field(
        default=32,
        metadata={
            "help": "Number of subprocesses to use for heavy use (tokenization)."
        },
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={
            "help": "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process."
        },
    )
    deepspeed: Optional[bool] = field(
        default=True,
        metadata={"help": "Use of deepspeed."},
    )

    deepspeed_config: Optional[str] = field(
        default=None,
        metadata={"help": "Deepspeed config file."},
    )

    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    disable_tqdm: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )

    lr_scheduler_type: Optional[SchedulerType] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )

    def __post_init__(self):
        if self.eval_steps is None:
            self.eval_steps = self.logging_steps

    
    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support).
        """
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d



    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)



    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        valid_types = [bool, int, float, str, torch.Tensor]
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}