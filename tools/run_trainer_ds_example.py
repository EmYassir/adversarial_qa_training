#!/usr/bin/env python
# coding=utf-8
import sys
import os
sys.path.insert(1, os.getcwd())

import torch
import random
import json
import math
import re
import deepspeed
import datetime

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from logging import (
    basicConfig, 
    getLogger, 
    StreamHandler
)

from datasets import (
    DatasetDict, 
    load_from_disk, 
    load_dataset
)
from transformers import (
    DataCollatorWithPadding,
    SchedulerType,
    get_scheduler,
    BertTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_utils import unwrap_model

from src.model.gan_modeling import FullGANModel
from src.config.gan_config import FullGANConfig
from src.utilities.trainer_utils import (
    compute_r_dis_loss,
    compute_gen_loss,
    get_collate_fn,
    get_generator_sample_mask,
    get_hits_dict,
)
from src.utilities.arguments import (
    ModelArguments,
    DataTrainingArguments,
    TrainingArguments,
)
from transformers.trainer_pt_utils import  IterableDatasetShard



def get_log_writer():
    basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[StreamHandler(sys.stdout)],
    )
    return getLogger(__name__)



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # Include DeepSpeed configuration arguments
    #parser = deepspeed.add_config_arguments(parser)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # logging
    logger = get_log_writer()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Similar to torch init_distributed
    deepspeed.init_distributed(dist_backend='nccl')
    
    # Loading the model
    if training_args.local_rank == 0:
        logger.info(f"===>>> Loading the config from {model_args.config_path}...")
    config = FullGANConfig.from_json_file(model_args.config_path)
    if training_args.local_rank == 0:
        logger.info(f"===>>> Loading the model from the configuration object...")
    
    model = FullGANModel(config).to(f"cuda:{training_args.local_rank}")
    generator = model.generator
    discriminator = model.discriminator
    ans_discriminator = model.ans_discriminator

    # Handle the repository creation
    if training_args.local_rank == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Wait for everyone
    torch.distributed.barrier()

    # Ensure that the hits list is a list of integers
    if model_args.hits_list is None:
        model_args.hits_list = [1, 3, 5, 10, 20, 30, 50]
    
    if training_args.local_rank == 0:
        logger.info(f"===>>> Loading the dataset from \'{data_args.dataset_path}\'...")
    
    # Get the datasets: Loads a processed dataset from disk. Must be a dataset created by a DatasetProcessor
    raw_datasets = load_from_disk(data_args.dataset_path)
    """
    # TODO remove once testing is over
    raw_datasets = DatasetDict(
        {
            "train": raw_datasets["train"].select(range(100)),
            "dev": raw_datasets["dev"].select(range(100)),
            "test": raw_datasets["test"].select(range(100)),
        }
    )
    """
    
    # Wait for everyone
    torch.distributed.barrier()

    #TODO: resize the model's length
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for submodel in [generator, discriminator, ans_discriminator]:
        submodel.bert_model.resize_token_embeddings(len(tokenizer))



    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 128:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 128
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Answerability labeling for each document
    def find_whole_answer(w):
        return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search

    def label_answerability(example):
        compile_func = find_whole_answer(
            "|".join([re.escape(answer) for answer in example["answer"]])
        )
        answerability_labels = list(
            map(bool, (map(compile_func, example["documents"])))
        )
        return {"answerability_labels": answerability_labels}

    if training_args.local_rank > 0:
        logger.info('Waiting for main process to perform the labeling')
        torch.distributed.barrier()
    
    logger.info(f"===>>> Labeling the dataset...")
    answerable_datasets = raw_datasets.map(
        label_answerability,
        num_proc=training_args.dataloader_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_rank > 0),
        desc="Labeling each document as having an answer or not",
    )

    # Signaling to everyone else that lkabeling is done
    if training_args.local_rank == 0:
        torch.distributed.barrier()

    # Keep only questions that have at least one answerable document
    answerable_datasets = answerable_datasets.filter(
        lambda example: any(example["answerability_labels"]),
        num_proc=training_args.dataloader_num_workers,
    )
    
    # Wait for everyone
    torch.distributed.barrier()

    # Tokenize the questions and documents
    def tokenize_function(examples):
        # Prepend each document with the question
        #TODO: take pad_to_max_length into account
        return tokenizer(
            [
                examples["question"] + tokenizer.sep_token + document
                for document in examples["documents"]
            ],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )



    if training_args.local_rank > 0:
        logger.info('Waiting for main process to perform the tokenization')
        torch.distributed.barrier()

    # Preprocessing the datasets.
    column_names = raw_datasets["train"].column_names
    tokenized_datasets = answerable_datasets.map(
        tokenize_function,
        num_proc=training_args.dataloader_num_workers,
        remove_columns=column_names,
        load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_rank > 0),
        desc="Running tokenizer on every text in dataset",
    )
    # Signaling to everyone else that tokenization is done
    if training_args.local_rank == 0:
        torch.distributed.barrier()

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["dev"]

    # Conditional for small test subsets
    """
    if len(train_dataset) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    """
    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = get_collate_fn(DataCollatorWithPadding(tokenizer))

    # DataLoaders creation
    if training_args.local_rank == 0:
        logger.info(f"===>>> Dataloader creation")

    train_dataloader = DataLoader(
        IterableDatasetShard(train_dataset) if isinstance(train_dataset, torch.utils.data.IterableDataset) else train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=int(training_args.dataloader_num_workers)
    )
    eval_dataloader = DataLoader(
        IterableDatasetShard(eval_dataset) if isinstance(eval_dataset, torch.utils.data.IterableDataset) else eval_dataset,
        collate_fn=data_collator,
        batch_size=training_args.per_device_eval_batch_size,
        num_workers=int(training_args.dataloader_num_workers)
    )
    # Wait for everyone
    torch.distributed.barrier()

    # Scheduler and math around the number of training steps.
    override_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_args.gradient_accumulation_steps
    )
    if training_args.max_steps <= 0:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        override_max_train_steps = True

    def init_optimizer(model):
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(grouped_parameters, lr=training_args.learning_rate)


        # Scheduler and math around the number of training steps.
        lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.num_warmup_steps,
            num_training_steps=training_args.max_train_steps,
        )
        return optimizer, lr_scheduler

    """
    # Initialize the optimizer for generator, discriminator, and answer discriminator
    gen_optimizer, gen_lr_scheduler = init_optimizer(generator)
    r_dis_optimizer, r_dis_lr_scheduler = init_optimizer(discriminator)
    a_dis_optimizer, a_dis_lr_scheduler = init_optimizer(ans_discriminator)
    """

    """
    generator_engine, gen_optimizer, _, gen_lr_scheduler = deepspeed.initialize(
        args=training_args,
        model=generator,
        model_parameters=generator.parameters(),
        optimizer=gen_optimizer,
        lr_scheduler=gen_lr_scheduler,
        config=training_args.deepspeed_config,
        dist_init_required = True
        )
    """
    generator_engine, _, _, _ = deepspeed.initialize(
        args=training_args,
        model=generator,
        model_parameters=generator.parameters(),
        config=training_args.deepspeed_config,
        dist_init_required = True
        )


    discriminator_engine, _, _, _ = deepspeed.initialize(
        args=training_args,
        model=discriminator,
        model_parameters=discriminator.parameters(),
        config=training_args.deepspeed_config,
        dist_init_required = False
        )
    ans_discriminator_engine, _, _, _ = deepspeed.initialize(
        args=training_args,
        model=ans_discriminator,
        model_parameters=ans_discriminator.parameters(),
        config=training_args.deepspeed_config,
        dist_init_required = False
        )
    
    # "Real" models
    generator = generator_engine.module
    discriminator = discriminator_engine.module
    ans_discriminator = ans_discriminator_engine.module
    
    # Wait for everyone
    torch.cuda.synchronize()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_args.gradient_accumulation_steps
    )
    if override_max_train_steps:
        training_args.max_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    if training_args.save_steps > 0:
        checkpointing_steps = int(training_args.save_steps)
    else:
        checkpointing_steps = None

    # TODO: tensorboard

    # Helper function for saving models
    def save_model(model, output_dir, local_rank):
        unwrap_model(model).save_pretrained(
            output_dir,
            is_main_process=(local_rank == 0),
        )

    # Train!
    total_batch_size = (
        training_args.per_device_train_batch_size
        * torch.distributed.get_world_size()
        * training_args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")

    # TODO: support resuming from checkpoint
    
    # Loss
    ans_loss_fn = torch.nn.BCEWithLogitsLoss()
    """
        # Save final model state
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_gan = accelerator.unwrap_model(gan)
            unwrapped_gan.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
    """

    completed_steps = 0
    starting_epoch = 0
    #progress_bar = tqdm(total=training_args.max_steps, disable=training_args.local_rank != 0)
    #progress_bar.set_description("Training steps")
    for epoch in range(starting_epoch, training_args.num_train_epochs):
        round_counter = 0
        for batch in tqdm(train_dataloader, disable=training_args.local_rank != 0):
            device_dsc = f"cuda:{training_args.local_rank}"
            batch = {key: value.to(device_dsc) for key, value in batch.items()}     
            for key, value in batch.items():
                print(f"batch[{key}].size() == {value.size()}")      
            with torch.cuda.amp.autocast():
                if round_counter < model_args.num_dis_rounds:
                    # Generator is in eval mode. Discriminators in train mode
                    generator.eval()
                    discriminator.train()
                    discriminator.zero_grad()
                    ans_discriminator.train()
                    ans_discriminator.zero_grad()

                    # Attempt to save some space
                    with torch.no_grad():
                        gen_scores = generator(**batch)["output_distribution"]
                    dis_scores = discriminator(**batch)["output_distribution"]
                    ans_scores = ans_discriminator(**batch)["output_distribution"]
                
                    # Begin a discriminator round
                    #true_dat_mask = batch["answerability_labels"].to(device_dsc)
                    true_dat_mask = batch["answerability_labels"].unsqueeze(0).to(device_dsc)
                    true_dis_scores = dis_scores[true_dat_mask]

                    gen_dat_mask = get_generator_sample_mask(batch, gen_scores)
                    gen_dis_scores = dis_scores[gen_dat_mask]
                    sample_gen_scores = gen_scores[gen_dat_mask]

                    # Compute losses
                    rank_loss = compute_r_dis_loss(
                        true_dis_scores,
                        gen_dis_scores,
                        true_dat_mask,
                        sample_gen_scores,
                    )
                    ans_loss = ans_loss_fn(ans_scores, true_dat_mask.float())

                    # Deepspeed requires special backward call
                    discriminator_engine.backward(rank_loss)
                    ans_discriminator_engine.backward(ans_loss)

                    # Deepspeed update of the optimizers
                    discriminator_engine.step()
                    ans_discriminator_engine.step()
                    
                    # progress_bar.update(1)
                    completed_steps += 1
                    round_counter += 1
                else:
                    # Generator is in train mode. Discriminators in eval mode
                    generator.train()
                    generator.zero_grad()
                    discriminator.eval()
                    ans_discriminator.eval()

                    gen_scores = generator(**batch)["output_distribution"]
                    
                    # Attempt to save some space
                    with torch.no_grad():
                        dis_scores = discriminator(**batch)["output_distribution"]
                        ans_scores = ans_discriminator(**batch)["output_distribution"]

                    loss = compute_gen_loss(
                        batch,
                        gen_scores,
                        dis_scores,
                        ans_scores,
                        model_args.ans_discriminator_weight,
                        model_args.regularizer_weight,
                    )

                    # Deepspeed backward/optimizer calls
                    generator_engine.backward(loss)
                    generator_engine.step()
                    #progress_bar.update(1)
                    completed_steps += 1

                    if round_counter >= model_args.num_dis_rounds + model_args.num_gen_rounds - 1:
                        round_counter = 0
                    else:
                        round_counter += 1
            
            if completed_steps % training_args.eval_steps == 0:

                """ Evaluate Discriminator """
                eval_metrics = get_eval_metrics(
                    model_args,
                    generator_engine.module,
                    discriminator_engine.module,
                    ans_discriminator_engine.module,
                    eval_dataloader,
                    device,
                    device,
                    device,
                )
                if training_args.local_rank == 0:
                    # TODO: log the training steps
                    logger.info(f"#### Evaluating ... Eval loss == {eval_metrics['eval_loss']}")
                    pass

            # Save the model
            if checkpointing_steps is not None and completed_steps % checkpointing_steps == 0:
                output_dir_suffix = f"model_epoch_{epoch}_chkpt_step_{completed_steps}"
                # TODO : Getting rid of extra model checkpoints, if need be
                if training_args.local_rank == 0:
                    output_dir = os.path.join(training_args.output_dir, output_dir_suffix)
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"#### Saving the model at {output_dir}")
                    save_model(model, rank = training_args.local_rank)
    
    torch.cuda.synchronize()
    # Save the final model
    
    if training_args.local_rank == 0:
        output_dir_suffix = f"model_final_step_{completed_steps}"
        logger.info(f"#### Saving the trained model at {output_dir_suffix}")
        output_dir = os.path.join(training_args.output_dir, output_dir_suffix)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"#### Saving the model at {output_dir}")
        save_model(model, rank = training_args.local_rank)

                
    


def get_eval_metrics(
    model_args,
    training_args,
    generator,
    discriminator,
    ans_discriminator,
    eval_dataloader,
    gen_device,
    dis_device,
    ans_device,
):
    # Prepare models for evaluation
    generator = generator.to(gen_device)
    discriminator = discriminator.to(dis_device)
    ans_discriminator = ans_discriminator.to(ans_device)
    generator.eval()
    discriminator.eval()
    ans_discriminator.eval()

    # Initialize the score lists
    losses = []
    all_gen_scores = []
    all_dis_scores = []
    all_ans_scores = []
    all_answerability_labels = []

    # Run through the dataloader
    for batch in eval_dataloader:
        device_dsc = f"cuda:{training_args.local_rank}"
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                batch = {key: value.to(device_dsc) for key, value in batch.items()}   
                gen_scores = generator(**batch)["output_distribution"]
                dis_scores = discriminator(**batch)["output_distribution"]
                ans_scores = ans_discriminator(**batch)["output_distribution"]
            loss = compute_gen_loss(
                batch,
                gen_scores,
                dis_scores,
                ans_scores,
                model_args.ans_discriminator_weight,
                model_args.regularizer_weight,
            )
        losses.append(loss.item())
        all_gen_scores.extend(gen_scores.tolist())
        all_dis_scores.extend(dis_scores.tolist())
        all_ans_scores.extend(ans_scores.tolist())
        all_answerability_labels.extend(batch["answerability_labels"].tolist())

    # For each model, calculate its hit score.
    # This is to monitor how each of the models performs in finding ground truth documents
    ret = {}
    ret.update(
        get_hits_dict(
            all_gen_scores, all_answerability_labels, model_args.hits_list, "generator"
        )
    )
    ret.update(
        get_hits_dict(
            all_dis_scores, all_answerability_labels, model_args.hits_list, "discriminator"
        )
    )
    ret.update(
        get_hits_dict(
            all_ans_scores,
            all_answerability_labels,
            model_args.hits_list,
            "ans_discriminator",
        )
    )
    eval_loss = sum(losses) / len(losses)
    ret["eval_loss"] = eval_loss
    return ret


if __name__ == "__main__":
    main()
