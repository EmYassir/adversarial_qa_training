import sys
import logging
from typing import Dict, Optional
from transformers import PretrainedConfig


## DEBUG ONLY
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


""" Configs for GAN's generator/discriminator models """


class GANConfig(PretrainedConfig):
    model_type = "bert_based_gan_config"

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def __init__(
        self,
        model_cfg: Optional[Dict] = None,
        model_name_or_path: Optional[str] = "bert-base-cased",
        initializer_range: float = 0.02,
        num_labels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if model_cfg is None and model_name_or_path is None:
            raise ValueError(
                f"Missing value(s): Neither of 'model_cfg' ({model_cfg}) nor 'model_name_or_path' ({model_name_or_path}) are provided."
            )
        self.model_cfg = model_cfg
        self.model_name_or_path = model_name_or_path
        self.initializer_range = initializer_range
        self.num_labels = num_labels


""" Config for the full GAN ensemble (generator and discriminator) """


class FullGANConfig(PretrainedConfig):
    model_type = "full_gan_config"

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def __init__(
        self,
        generator_cfg: Dict = None,
        discriminator_cfg: Dict = None,
        ans_discriminator_cfg: Dict = None,
        gen_rounds: int = 6,
        dis_rounds: int = 8,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.generator_cfg = generator_cfg
        self.discriminator_cfg = discriminator_cfg
        self.ans_discriminator_cfg = ans_discriminator_cfg
        self.gen_rounds = gen_rounds
        self.dis_rounds = dis_rounds
        self.initializer_range = initializer_range
