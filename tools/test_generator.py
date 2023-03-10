import os
import sys

sys.path.insert(1, os.getcwd())


import torch
import argparse
import logging
import shutil

from transformers import BertModel

from src.config.gan_config import GANConfig
from src.model.gan_modeling import DefaultModel


from transformers import BertTokenizer


def cleanup_output_dirs(dirpath):
    try:
        shutil.rmtree(dirpath, ignore_errors=True)
    except OSError as error:
        logger.info(error)
        logger.info(f"!!! Directory {dirpath} can not be removed !!!")


def test_1(args):
    model = None
    logger.info(f"===>>> Loading the config from {args.config_path}...")
    config = GANConfig.from_json_file(args.config_path)
    logger.info(f"===>>> Loading the model from the configuration object...")
    model = DefaultModel(config)
    if args.device is not None:
        logger.info(f"===>>> Moving model to device '{args.device}'...")
        model = model.move_to(f"cuda:{args.device}")
    logger.info(f"===>>> Moving the model back to cpu ...")
    model = model.cpu()
    assert model is not None


def test_2(args):
    model = None
    logger.info(f"===>>> Loading the config from {args.config_path}...")
    config = GANConfig.from_json_file(args.config_path)
    logger.info(f"===>>> Loading the model from the configuration object...")
    model = DefaultModel(config)
    if args.device is not None:
        logger.info(f"===>>> Moving model to device '{args.device}'...")
        model = model.move_to(f"cuda:{args.device}")
    assert model is not None
    logger.info(f"===>>> Saving the weights at {args.output_dir}...")
    model.save_to_disk(args.output_dir)
    assert os.path.isfile(os.path.join(args.output_dir, "config.json"))
    assert os.path.isfile(os.path.join(args.output_dir, "pytorch_model.bin"))
    logger.info(f"===>>> Moving the model back to cpu ...")
    model = model.cpu()
    logger.info(f"===>>> Loading the weights from {args.output_dir}...")
    model = DefaultModel.load_from_disk(args.output_dir)
    # model = GeneratorModel.from_pretrained(args.output_dir)
    assert model is not None
    assert model.bert_model is not None
    assert model.device == torch.device("cpu")
    logger.info(f"===>>> Moving the model back to cpu ...")
    model = model.cpu()

    logger.info(f"===>>> Cleanup ...")
    cleanup_output_dirs(args.output_dir)
    logger.info(f"===>>> DONE")


def test_3(args):
    logger.info(f"===>>> Loading the tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = None
    logger.info(f"===>>> Loading the config from {args.config_path}...")
    config = GANConfig.from_json_file(args.config_path)
    logger.info(f"===>>> Loading the model from the configuration object...")
    model = DefaultModel(config)
    assert model is not None
    if args.device is not None:
        logger.info(f"===>>> Moving model to device '{args.device}'...")
        model = model.move_to(f"cuda:{args.device}")

    logger.info(f"===>>> Testing forward passes ...")
    text_list = []
    for _ in range(100):
        text_list.append(
            'Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),'
            "to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell "
            "copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition"
        )
    encoding = tokenizer(
        text_list,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_attention_mask=True,
        is_split_into_words=False,
        return_tensors="pt",
    ).to(f"cuda:{args.device}")

    # To avoid out-of-memory errors, we use eval mode
    #model.train()
    model.eval()
    with torch.no_grad():
        logger.info(
            f"===>>> Pass 1: k_select = {int(len(text_list)/2)}, n_documents = {len(text_list)}"
        )
        _ = model(
            **encoding, n_documents=len(text_list), k_select=int(len(text_list) / 2)
        )
        logger.info(
            f"===>>> Pass 2: k_select = {len(text_list)}, n_documents = {len(text_list)}"
        )
        _ = model(**encoding, n_documents=len(text_list), k_select=len(text_list))
        logger.info(f"===>>> Pass 3: k_select = {0}, n_documents = {len(text_list)}")
        _ = model(**encoding, n_documents=len(text_list), k_select=0)

    logger.info(f"===>>> Saving the weights at {args.output_dir}...")
    model.save_to_disk(args.output_dir)
    assert os.path.isfile(os.path.join(args.output_dir, "config.json"))
    assert os.path.isfile(os.path.join(args.output_dir, "pytorch_model.bin"))
    logger.info(f"===>>> Moving the model back to cpu ...")
    model = model.cpu()

    logger.info(f"===>>> Loading the weights from {args.output_dir}...")
    model = DefaultModel.load_from_disk(args.output_dir)
    assert model.bert_model is not None
    assert model.device == torch.device("cpu")
    logger.info(f"===>>> Moving the model back to cpu ...")
    model = model.cpu()

    logger.info(f"===>>> Cleanup ...")
    cleanup_output_dirs(args.output_dir)
    logger.info(f"===>>> DONE")


def test_4(args):
    model_name = "bert-base-cased"
    model = None
    logger.info(f"===>>> Loading the config from {args.config_path}...")
    config = GANConfig.from_json_file(args.config_path)
    config.model_name_or_path = "bert-base-cased"
    logger.info(f"===>>> Loading the model from config pointing to '{model_name}'...")

    model = DefaultModel(config)
    assert model is not None
    if args.device is not None:
        logger.info(f"===>>> Moving model to device '{args.device}'...")
        model = model.move_to(f"cuda:{args.device}")

    logger.info(f"===>>> Saving the model to '{args.output_dir}'...")
    model.save_to_disk(args.output_dir)
    assert os.path.isfile(os.path.join(args.output_dir, "config.json"))
    assert os.path.isfile(os.path.join(args.output_dir, "pytorch_model.bin"))

    logger.info(f"===>>> Loading the weights from {args.output_dir}...")
    model = DefaultModel.load_from_disk(args.output_dir)
    assert model.bert_model is not None
    assert model.device == torch.device("cpu")
    logger.info(f"===>>> Moving the model back to cpu ...")
    model = model.cpu()

    logger.info(f"===>>> Cleanup ...")
    cleanup_output_dirs(args.output_dir)
    logger.info(f"===>>> DONE")


def test_5(args):
    model_name = "bert-base-cased"
    model_path = os.path.join(args.output_dir, "bert_hf")
    os.makedirs(model_path, exist_ok=True)

    logger.info(f"===>>> Downloading and saving '{model_name}' to '{model_path}' ...")
    model = BertModel.from_pretrained(model_name)
    model.save_pretrained(model_path)

    model = None
    logger.info(f"===>>> Loading the config from {args.config_path}...")
    config = GANConfig.from_json_file(args.config_path)
    config.model_name_or_path = model_path

    logger.info(f"===>>> Loading the model from config pointing to '{model_path}'...")
    model = DefaultModel(config)
    assert model is not None
    if args.device is not None:
        logger.info(f"===>>> Moving model to device '{args.device}'...")
        model = model.move_to(f"cuda:{args.device}")
    logger.info(f"===>>> Saving the model to to '{args.output_dir}'...")
    model.save_to_disk(args.output_dir)
    assert os.path.isfile(os.path.join(args.output_dir, "config.json"))
    assert os.path.isfile(os.path.join(args.output_dir, "pytorch_model.bin"))

    logger.info(f"===>>> Loading the weights from {args.output_dir}...")
    model = DefaultModel.load_from_disk(args.output_dir)
    assert model.bert_model is not None
    assert model.device == torch.device("cpu")
    logger.info(f"===>>> Moving the model back to cpu ...")
    model = model.cpu()

    logger.info(f"===>>> Cleanup ...")
    cleanup_output_dirs(args.output_dir)
    logger.info(f"===>>> DONE")


if __name__ == "__main__":
    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    # Parser
    parser = argparse.ArgumentParser(description="Testing GAN components")
    parser.add_argument(
        "--option", required=True, choices=["generator", "discriminator"], type=str
    )
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--device", default=None, type=int)
    args = parser.parse_args()

    logger.info(f"######## Testing the {args.option} ########")

    logger.info(f"## Test 1: Loading the model from config file...")
    test_1(args)

    logger.info(f"## Test 2: Saving the weights and reloading them...")
    test_2(args)

    logger.info(f"## Test 3: Testing the forward pass...")
    test_3(args)

    logger.info(f"## Test 4: Testing model load with HF pretrained bert...")
    test_4(args)

    logger.info(f"## Test 5: Testing model load with saved HF pretrained bert...")
    test_5(args)

    logger.info("DONE")
