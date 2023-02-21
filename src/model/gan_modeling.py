import os
import sys
import torch
import logging
from torch import nn as nn
from dataclasses import dataclass


from typing import (
    Union,
    Tuple,
    Optional,
    Callable
)

from torch import sigmoid
from torch import softmax

from transformers import  BertConfig, BertForSequenceClassification, AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling
from transformers.utils import ModelOutput

from src.config.gan_config import GANConfig, FullGANConfig
from src.utilities.trainer_utils import sample_from_tensor


## DEBUG ONLY
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)



@dataclass
class FullGANOutput(ModelOutput):
    """
    Class for outputs of the full GAN model.
    Args:
        generator_output (ModelOutput):
            The output of the generator.
        discriminator_output (ModelOutput):
            The output of the generator.
        ans_discriminator_output (ModelOutput):
            The output of the generator.
    """

    generator_output: Optional[SequenceClassifierOutput] = None
    discriminator_output: Optional[SequenceClassifierOutput] = None
    ans_discriminator_output: Optional[SequenceClassifierOutput] = None


"""""" """""" """""" """""" """""" """""
    Main GAN models' interfaces 
""" """""" """""" """""" """""" """""" ""

class GANModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization.
    """

    config_class = GANConfig
    load_tf_weights = None
    base_model_prefix = "bert_based_gan_model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Override main method to avoid some issues
    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights should be skipped when not initializing all weights
        # since from_pretrained(...) calls tie weights anyways
        self.tie_weights()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value

    def move_to(self, device):
        return self.to(device)

    @classmethod
    def load_from_disk(cls, dir_path):
        return cls.from_pretrained(dir_path)

    def save_to_disk(self, dir_path):
        # Save the model
        self.save_pretrained(dir_path)


class DefaultModel(GANModel):
    base_model_prefix = "bert_based_gan_model"

    def __init__(self, config: GANConfig):
        super().__init__(config)

        # Intialize simple fields
        self.model_name_or_path = config.model_name_or_path

        # Loading the model
        if self.model_name_or_path is not None and len(self.model_name_or_path) > 0:
            """
            self.bert_model = BertForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels = config.num_labels
            )
            """
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels = config.num_labels
            )
        else:
            config = BertConfig.from_dict(config.model_cfg)
            #self.bert_model = BertForSequenceClassification(config, config.num_labels)
            self.bert_model = AutoModelForSequenceClassification(config, config.num_labels)
            if self.bert_model.config.hidden_size <= 0:
                raise ValueError(
                    f"Encoder hidden_size ({self.bert_model.config.hidden_size}) should be positive"
                )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.IntTensor,
        attention_mask: Optional[torch.IntTensor] = None,
        token_type_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.FloatTensor, ...]]:

        # Reshape the batch (batch, n_docs, seq_len) -> (batch * n_docs, seq_len)
        input_ids = (
            input_ids.reshape((input_ids.size(0) * input_ids.size(1), -1))
            if input_ids is not None and input_ids.ndim > 2
            else input_ids
        )

        attention_mask = (
            attention_mask.reshape((attention_mask.size(0) * attention_mask.size(1), -1))
            if attention_mask is not None and attention_mask.ndim > 2
            else attention_mask
        )

        token_type_ids = (
            token_type_ids.reshape((token_type_ids.size(0) * token_type_ids.size(1), -1))
            if token_type_ids is not None and token_type_ids.ndim > 2
            else token_type_ids
        )
        inputs_embeds = (
            inputs_embeds.reshape((inputs_embeds.size(0) * inputs_embeds.size(1), -1))
            if (inputs_embeds is not None and inputs_embeds.ndim > 2)
            else inputs_embeds
        )

        return self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @property
    def embeddings_size(self) -> int:
        return self.bert_model.config.hidden_size

    def move_to(self, device):
        return self.to(device)



"""""" """""" """""" """""" """
    FULL GAN implementation 
""" """""" """""" """""" """"""
class FullGANInterface(PreTrainedModel):
    """
    An abstract class to handle weights initialization.
    """

    config_class = FullGANConfig
    load_tf_weights = None
    base_model_prefix = "full_gan_interface"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        # We have some nested pretrained modules...
        if isinstance(module, PreTrainedModel):
            pass  # These modules have been initialized already
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Override main method to avoid some issues
    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights should be skipped when not initializing all weights
        # since from_pretrained(...) calls tie weights anyways
        self.tie_weights()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value

    @classmethod
    def load_from_disk(cls, dir_path):
        return cls.from_pretrained(dir_path)

    def save_to_disk(self, dir_path):
        # Save the model
        self.save_pretrained(dir_path)


""" Full gan framework """
class FullGANModel(FullGANInterface):
    base_model_prefix = "full_gan_model"

    def __init__(self, config: FullGANConfig):
        super().__init__(config)
        #self.config = config

        # Generator
        subconfig = GANConfig.from_dict(self.config.generator_cfg)
        self.generator = DefaultModel(subconfig)
        subconfig = GANConfig.from_dict(self.config.discriminator_cfg)
        self.discriminator = DefaultModel(subconfig)
        subconfig = GANConfig.from_dict(self.config.ans_discriminator_cfg)
        self.ans_discriminator = DefaultModel(subconfig)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.IntTensor,
        attention_mask: Optional[torch.IntTensor] = None,
        token_type_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        options: str = "gad",
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.FloatTensor, ...]]:
        ret = {}
        # 'g' for Generator
        if "g" in options or "G" in options:
            ret["generator_output"] = self.generator(
                input_ids,
                attention_mask,
                token_type_ids,
                inputs_embeds,
                output_attentions,
                output_hidden_states,
                return_dict
            )
        # 'd' for Discriminator
        if "d" in options or "D" in options:
            ret["discriminator_output"] = self.discriminator(
                input_ids,
                attention_mask,
                token_type_ids,
                inputs_embeds,
                output_attentions,
                output_hidden_states,
                return_dict,
            )

        # 'a' for Answwer-discriminator
        if "a" in options or "A" in options:
            ret["ans_discriminator_output"] = self.ans_discriminator(
                input_ids,
                attention_mask,
                token_type_ids,
                inputs_embeds,
                output_attentions,
                output_hidden_states,
                return_dict,
            )

        if not return_dict:
            tuple_ret = ()
            for key in [
                "generator_output",
                "discriminator_output",
                "ans_discriminator_output",
            ]:
                if key in ret:
                    tuple_ret = (tuple_ret, ret[key])
            return tuple_ret

        return FullGANOutput(
            generator_output=ret["generator_output"]
            if "generator_output" in ret
            else None,
            discriminator_output=ret["discriminator_output"]
            if "discriminator_output" in ret
            else None,
            ans_discriminator_output=ret["ans_discriminator_output"]
            if "ans_discriminator_output" in ret
            else None,
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, *model_args, **kwargs):
        config = GANConfig.from_json_file(os.path.join(pretrained_model_path, "config.json"))
        config.generator_cfg.model_name_or_path = os.path.join(pretrained_model_path, "gen")
        config.discriminator_cfg.model_name_or_path = os.path.join(pretrained_model_path, "dis")
        config.ans_discriminator_cfg.model_name_or_path = os.path.join(pretrained_model_path, "ans")
        return cls(config)


    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "10GB",
        **kwargs,
    ):
        self.config.to_json_file(os.path.join(save_directory, "config.json"), use_diff=True)
        self.generator.save_pretrained(os.path.join(save_directory, "gen"), is_main_process, state_dict, save_function, push_to_hub, max_shard_size, **kwargs)
        self.discriminator.save_pretrained(os.path.join(save_directory, "dis"), is_main_process, state_dict, save_function, push_to_hub, max_shard_size, **kwargs)
        self.ans_discriminator.save_pretrained(os.path.join(save_directory, "ans"), is_main_process, state_dict, save_function, push_to_hub, max_shard_size, **kwargs)


    @property
    def embeddings_size(self) -> int:
        return self.generator.bert_model.config.hidden_size

    def move_to(self, device):
        return self.to(device)
