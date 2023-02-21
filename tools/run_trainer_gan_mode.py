#!/usr/bin/env python
# coding=utf-8
import sys
import os
sys.path.insert(1, os.getcwd())

import torch
import math

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from logging import (
    basicConfig, 
    getLogger, 
    StreamHandler
)

from datasets import load_from_disk
from transformers import (
    DataCollatorWithPadding,
    BertTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from src.model.gan_modeling import FullGANModel
from src.config.gan_config import FullGANConfig
from src.utilities.trainer_utils import (
    calculate_gen_dis_steps,
    verify_output_directory,
    compute_ans_loss,
    compute_r_dis_loss,
    compute_gen_loss,
    get_collate_fn,
    compute_device_map,
    sample_from_tensor,
    evaluate_and_log,
    clip_grad_norm,
    init_optimizer,
    label_datasets,
    tokenize_datasets,
    step,
    checkpoint
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

    # Ensure that the hits list is a list of integers
    if model_args.hits_list is None:
        model_args.hits_list = [1, 3, 5, 10, 20, 30, 50]
    
    # Ensure that the device map is set
    if model_args.device_map is None:
        model_args.device_map = compute_device_map()
    else:
        model_args.device_map = compute_device_map(devices = model_args.device_map)
    

    # Initialize the distributed environment
    #torch.init_distributed(dist_backend='nccl')

    # Preparing the output directories
    if training_args.local_rank in {-1, 0}:
        verify_output_directory(training_args)

    # Wait for everyone
    #torch.distributed.barrier()
    
    # Loading the model
    if training_args.local_rank in {-1, 0}:
        logger.info(f"===>>> Loading the config from {model_args.config_path}...")
    config = FullGANConfig.from_json_file(model_args.config_path)
    if training_args.local_rank in {-1, 0}:
        logger.info(f"===>>> Loading the model from the configuration object...")
    
    model = FullGANModel(config)
    generator = model.generator.to(f"cuda:{model_args.device_map['generator'][training_args.local_rank]}")
    discriminator = model.discriminator.to(f"cuda:{model_args.device_map['discriminator'][training_args.local_rank]}")
    ans_discriminator = model.ans_discriminator.to(f"cuda:{model_args.device_map['ans_discriminator'][training_args.local_rank]}")


    # Handle the repository creation
    if training_args.local_rank  in {-1, 0}:
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Wait for everyone
    #torch.distributed.barrier()

    if training_args.local_rank in {-1, 0}:
        logger.info(f"===>>> Loading the dataset from \'{data_args.dataset_path}\'...")
    
    # Get the datasets: Loads a processed dataset from disk. Must be a dataset created by a DatasetProcessor
    raw_datasets = load_from_disk(data_args.dataset_path)

    
    # Wait for everyone
    #torch.distributed.barrier()

    #TODO: resize the model's length
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for submodel in [generator, discriminator, ans_discriminator]:
        submodel.bert_model.resize_token_embeddings(len(tokenizer))


    if data_args.max_seq_length is None:
        data_args.max_seq_length = tokenizer.model_max_length
        if data_args.max_seq_length > 256:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 256 instead. You can change that default value by passing --max_seq_length xxx."
            )
            data_args.max_seq_length = 256
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        data_args.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)


    # Keep only questions that have at least one answerable document
    answerable_datasets = label_datasets(training_args, raw_datasets, logger, data_args.overwrite_cache)
    
    # Tokenizing the datasets
    tokenized_datasets = tokenize_datasets(
        data_args,
        training_args, 
        answerable_datasets, 
        raw_datasets["train"].column_names,  
        tokenizer, 
        logger, 
        data_args.overwrite_cache
        )
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["dev"]


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
    #torch.distributed.barrier()

    # Scheduler and math around the number of training steps.
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_steps is None or training_args.max_steps <= 0:
        training_args.max_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)

    # Initialize the optimizer for generator, discriminator, and answer discriminator
    gen_optimizer, gen_lr_scheduler = init_optimizer(generator, model_args.num_gen_rounds)
    r_dis_optimizer, r_dis_lr_scheduler = init_optimizer(discriminator, model_args.num_dis_rounds)
    a_dis_optimizer, a_dis_lr_scheduler = init_optimizer(ans_discriminator, model_args.num_dis_rounds)
    
    # Wait for everyone
    #torch.cuda.synchronize()

    # Initialize tensorboard loggers
    writer_dis = SummaryWriter(log_dir=training_args.logging_dir + f"/discriminator")
    writer_ans = SummaryWriter(log_dir=training_args.logging_dir + f"/ans_discriminator")
    writer_gen = SummaryWriter(log_dir=training_args.logging_dir + f"/generator")
    writers_dic = {"gen": writer_dis, "dis": writer_dis, "ans": writer_ans}

    # Train!
    total_batch_size = (
        training_args.per_device_train_batch_size
        #* torch.distributed.get_world_size() Uncomment for distributed training
        * training_args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")

    d_max_steps, g_max_steps = calculate_gen_dis_steps(training_args.max_steps, model_args.num_dis_rounds, model_args.num_gen_rounds)
    logger.info(f"  Total optimization steps for the generator = {g_max_steps}")
    logger.info(f"  Total optimization steps for the discriminators = {d_max_steps}")

    # TODO: support resuming from checkpoint
    completed_steps = 0
    completed_steps_gen = 0
    completed_steps_dis = 0
    starting_epoch = 0
    
    #progress_bar = tqdm(total=training_args.max_steps, disable=training_args.local_rank != 0)
    #progress_bar.set_description("Training steps")
    device_map, local_rank = model_args.device_map, training_args.local_rank
    scaler = torch.cuda.amp.GradScaler()

    # This allows to alternate between generator and discriminators
    round_counter = 0
    for epoch in range(starting_epoch, training_args.num_train_epochs):
        for batch in tqdm(train_dataloader, disable=training_args.local_rank  not in {0, -1}): 
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
                        gen_scores = generator(**{key: value.to(f"cuda:{device_map['generator'][local_rank]}") for key, value in batch.items()})["output_distribution"]
                    dis_scores = discriminator(**{key: value.to(f"cuda:{device_map['discriminator'][local_rank]}") for key, value in batch.items()})["output_distribution"]
                    ans_scores = ans_discriminator(**{key: value.to(f"cuda:{device_map['ans_discriminator'][local_rank]}") for key, value in batch.items()})["output_distribution"]
                
                    # Begin a discriminator round
                    true_dat_mask = batch["answerability_labels"].unsqueeze(0).to(f"cuda:{device_map['discriminator'][local_rank]}")
                    true_dis_scores = dis_scores[true_dat_mask]

                    gen_dat_mask, sample_gen_scores = sample_from_tensor(gen_scores, training_args.n_samples, with_replacement = False)
                    gen_dis_scores = dis_scores[gen_dat_mask.cpu()]

                    # Compute losses
                    rank_loss = compute_r_dis_loss(
                        true_dis_scores,
                        gen_dis_scores,
                        true_dat_mask,
                        sample_gen_scores.to(gen_dis_scores.device),
                    )
                    ans_loss = compute_ans_loss(ans_scores, true_dat_mask.float().to(ans_scores.device))

                    # Scaling
                    ans_loss = ans_loss / training_args.gradient_accumulation_steps
                    rank_loss = rank_loss / training_args.gradient_accumulation_steps

                    # Backward loss
                    scaler.scale(rank_loss).backward()
                    scaler.scale(ans_loss).backward()


                    # progress_bar.update(1)
                    completed_steps += 1
                    completed_steps_dis += 1
                    round_counter += 1

                    # Logging 
                    if completed_steps_dis % training_args.logging_steps == 0:
                        writer_ans.add_scalar('ans_loss', ans_loss.item(), completed_steps_dis)
                        writer_dis.add_scalar('dis_loss', rank_loss.item(), completed_steps_dis)

                    # Updates the scale for next iteration.
                    if (
                        completed_steps_dis % training_args.gradient_accumulation_steps == 0
                        or completed_steps == len(train_dataloader) - 1
                    ):
                        # Gradient norm clipping
                        clip_grad_norm(
                            training_args.max_grad_norm, 
                            scaler, 
                            [discriminator, ans_discriminator], 
                            [r_dis_optimizer, a_dis_optimizer])

                        # step() with scaler/optimizers/schedulers
                        step(scaler, [r_dis_optimizer, a_dis_optimizer], [r_dis_lr_scheduler, a_dis_lr_scheduler])

                else:
                    # Generator is in train mode. Discriminators in eval mode
                    generator.train()
                    generator.zero_grad()
                    discriminator.eval()
                    ans_discriminator.eval()

                    gen_scores = generator(**{key: value.to(f"cuda:{device_map['generator'][local_rank]}") for key, value in batch.items()})["output_distribution"]
                    
                    # Attempt to save some space
                    with torch.no_grad():
                        dis_scores = discriminator(**{key: value.to(f"cuda:{device_map['discriminator'][local_rank]}") for key, value in batch.items()})["output_distribution"]
                        ans_scores = ans_discriminator(**{key: value.to(f"cuda:{device_map['ans_discriminator'][local_rank]}") for key, value in batch.items()})["output_distribution"]

                    loss = compute_gen_loss(
                        batch,
                        gen_scores,
                        dis_scores,
                        ans_scores,
                        model_args.ans_discriminator_weight,
                        model_args.regularizer_weight,
                    )

                    # Deepspeed backward/optimizer calls
                    scaler.scale(loss).backward()
                    
                    #progress_bar.update(1)
                    completed_steps += 1
                    completed_steps_gen += 1

                    # Logging
                    if completed_steps_gen % training_args.logging_steps == 0:
                        writer_gen.add_scalar('gen_loss', loss.item(), completed_steps_gen)

                     # Updates the scale for next iteration.
                    if (
                        completed_steps_gen % training_args.gradient_accumulation_steps == 0
                        or completed_steps == len(train_dataloader) - 1
                    ):                    
                        # Gradient norm clipping
                        clip_grad_norm(training_args.max_grad_norm, scaler, [generator], [gen_optimizer])

                        # step() with scaler/optimizers/schedulers
                        step(scaler, [gen_optimizer], [gen_lr_scheduler])

                    if round_counter >= model_args.num_dis_rounds + model_args.num_gen_rounds - 1:
                        round_counter = 0
                    else:
                        round_counter += 1
            
            # Evaluation and logging to tensorboard
            evaluate_and_log(model_args, training_args, completed_steps, model, eval_dataloader, logger,  writers = writers_dic, model_name = None)
            
            # Save the model
            checkpoint(training_args, completed_steps, model, epoch, logger, False)
    
    
    #torch.cuda.synchronize()
    # Save the final model
    checkpoint(training_args, completed_steps, model, epoch, logger, True)


if __name__ == "__main__":
    main()
