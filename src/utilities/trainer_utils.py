import os
import sys
import torch
import logging
import shutil
import re
from tqdm.auto import tqdm
from functools import partial
import random
import pickle as pkl
from torch.utils.data import DataLoader

from transformers import get_scheduler
from transformers.modeling_utils import unwrap_model
from transformers.trainer_pt_utils import  IterableDatasetShard

from sklearn.metrics import ndcg_score
from datasets import DatasetDict

# DEBUG
import uuid




## DEBUG ONLY
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def verify_output_directory(args):
    output_dir = args.output_dir
    if args.overwrite_output_dir == True:
        remove_dir(output_dir)
        os.makedirs(output_dir, exist_ok = True)
    elif (os.path.isdir(output_dir) == True) and (len(os.listdir(output_dir)) > 0):
        raise OSError(f"Folder \'{output_dir}\' exists and is non empty. Please use the option \'--overwrite_output_dir\' to override it.")


def get_index_list(batch_field, indices):
    return torch.stack(
        [batch_field[i][j] for i, index_list in enumerate(indices) for j in index_list]
    )

def compute_gen_loss(
    batch,
    gen_scores,
    dis_scores,
    ans_scores,
    num_gen_samples=50,
    ans_dis_weight=0.25,
    reg_weight=1,
    lambda_val=0.5,
):
    """
    answerability_mask = batch["answerability_labels"].to(gen_scores.device)

    # Clamp scores since we will perform log(gen_scores)
    soft_gen_scores = torch.softmax(gen_scores, dim=0).clamp(min=1e-8)

    # For each batch of questions, Find the uniform probability of picking a ground truth document
    # prob_true = P(d), d ~ p_true, for each question
    prob_true = 1.0 / torch.sum(answerability_mask)

    regularizer = torch.sum(
        prob_true
        * answerability_mask
        * (-torch.log(soft_gen_scores) + torch.log(prob_true) * answerability_mask)
    )
    # gen_dat_mask = get_generator_sample_mask(gen_batch, gen_scores, num_gen_samples)
    gen_labels = (sample_from_tensor(soft_gen_scores, num_gen_samples)[0]).to(soft_gen_scores.device)

    # Importance sampling
    prob_IS = soft_gen_scores * (1.0 - lambda_val)
    prob_IS = prob_IS + answerability_mask * lambda_val * prob_true
    choose_IS = (soft_gen_scores / prob_IS) * gen_labels

    # Convert logits into scores
    # Place both scores in generator device
    ans_scores = torch.sigmoid(ans_scores).to(gen_scores.device)
    dis_scores = torch.sigmoid(dis_scores).to(gen_scores.device)

    loss = -torch.mean(
        torch.log(soft_gen_scores) 
        * (torch.sigmoid(dis_scores).to(gen_scores.device) 
        + ans_dis_weight * torch.sigmoid(ans_scores).to(gen_scores.device)) 
        * choose_IS
    )
    return loss + reg_weight * regularizer
    """
    ######### OUSSAMA IMPLEMENTATION HERE
    answerability_mask = batch["answerability_labels"]

    # Clamp scores since we will perform log(gen_scores)
    gen_scores = torch.softmax(gen_scores, dim=0).clamp(min=1e-8)

    # For each batch of questions, Find the uniform probability of picking a ground truth document
    # prob_true = P(d), d ~ p_true, for each question
    prob_true = 1.0 / torch.sum(answerability_mask)

    regularizer = torch.sum(
        prob_true
        * answerability_mask
        * (-torch.log(gen_scores) + torch.log(prob_true) * answerability_mask)
    )
    
    #gen_dat_indices = sample_from_tensor(gen_scores, 250, with_replacement=True)[0].to(gen_scores.device)
    gen_dat_indices = torch.multinomial(gen_scores, 250, replacement=True)

    # Importance sampling
    prob_IS = gen_scores * (1.0 - lambda_val)
    prob_IS = prob_IS + answerability_mask * lambda_val * prob_true
    choose_IS = gen_scores / prob_IS
    choose_IS = choose_IS[gen_dat_indices]

    # Convert logits into scores

    # Place both scores in generator device
    ans_scores = torch.sigmoid(ans_scores).to(gen_scores.device)[gen_dat_indices]
    dis_scores = torch.sigmoid(dis_scores).to(gen_scores.device)[gen_dat_indices]

    #gen_dat_indices = torch.multinomial(gen_scores, 250, replacement=True)
    gen_scores = gen_scores[gen_dat_indices]

    loss = -torch.mean(
        torch.log(gen_scores) * (dis_scores + ans_dis_weight * ans_scores[gen_dat_indices]) * choose_IS
    )
    return loss + reg_weight * regularizer


def compute_ans_loss(ans_scores, ans_batch):
    # Calculate weight for positive examples due to class imbalance
    """
    batch_size = ans_batch["input_ids"].shape[0]
    num_docs = ans_batch["input_ids"].shape[1]
    w0 = batch_size * num_docs / (ans_batch["answerability_labels"].sum())

    ans_loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([w0]).to(ans_scores.device)
    )
    """
    ans_loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = ans_loss_fn(ans_scores, ans_batch["answerability_labels"].float())    
    return loss


def compute_r_dis_loss(batch, dis_scores, gen_scores, n_samples):
    ############### STARTS HERE
    """
    rank_loss_fn = torch.nn.BCEWithLogitsLoss()

    # Get discriminator scores for ground truth documents
    # Set the labels to 1
    ans_labels = batch["answerability_labels"].to(dis_scores.device)


    # Select documents from the generator
    # Set the labels to 0
    gen_labels = (sample_from_tensor(torch.softmax(gen_scores, dim=0).clamp(min=1e-8), n_samples)[0]).to(dis_scores.device)

    # Compute loss of discriminator
    #rank_loss = rank_loss_fn(dis_scores, ans_labels) + rank_loss_fn(dis_scores, torch.logical_not(gen_labels).long())
    rank_loss = rank_loss_fn(dis_scores, ans_labels * torch.logical_not(gen_labels).float())
    """
    ############### OUSSAMA CODE HERE
    rank_loss_fn = torch.nn.BCEWithLogitsLoss()

    # Get discriminator scores for ground truth documents
    # Set the labels to 1

    answerability_labels = batch["answerability_labels"]
    pos_dis_scores = dis_scores[answerability_labels]
    pos_dis_labels = torch.ones_like(pos_dis_scores)

    # Select documents from the generator
    # Set the labels to 0

    #gen_dat_mask = get_generator_sample_mask(batch, gen_scores)
    #gen_dat_mask = (sample_from_tensor(torch.softmax(gen_scores, dim=0).clamp(min=1e-8), n_samples)[0]).bool().to(dis_scores.device)
    #gen_dat_mask = (sample_from_tensor(torch.softmax(gen_scores, dim=0).clamp(min=1e-8), int(answerability_labels.sum()))[0]).bool().to(dis_scores.device)
    gen_dat_mask = get_generator_sample_mask(batch, gen_scores)
    neg_dis_scores = dis_scores[gen_dat_mask]
    neg_dis_labels = torch.zeros_like(neg_dis_scores)

    # Compute loss of discriminator
    rank_loss = rank_loss_fn(neg_dis_scores, neg_dis_labels) + rank_loss_fn(
        pos_dis_scores, pos_dis_labels
    )
    return rank_loss

# Debug function
def is_abnormal(dic, models_names, hits = [1, 3, 5, 10, 20, 30, 50]):
  for model_name in models_names:
    min_value = -sys.maxsize - 1
    for hit in hits:
      key = model_name + "_hits@" + str(hit)
      if key not in dic:
        continue
      value = dic[key]
      if value < min_value:
        return True
      min_value = value
  return False

def compute_hits(
    all_scores, all_answerability_labels, hits_list=[1, 3, 5, 10, 20, 30, 50]
):
    ndcg_gen_hits = [ndcg_score(all_answerability_labels, all_scores, k=hits) for hits in hits_list]
    return ndcg_gen_hits


def get_hits_dict(all_scores, all_answerability_labels, hits_list, model_name):
    hits_scores = compute_hits(all_scores, all_answerability_labels, hits_list)
    return {
        model_name + "_hits@" + str(hit): round(h_score, 4)
        for hit, h_score in zip(hits_list, hits_scores)
    }


def get_generator_sample_mask(batch, scores):
    mask = batch["answerability_labels"]

    out_mask = torch.zeros(mask.shape, dtype=torch.bool)

    num_to_sample = int(mask.sum())

    # Sample as many generator-chosen documents as there are answerable documents
    row_indices = torch.multinomial(
        torch.softmax(scores, 0), num_to_sample, replacement=False
    )

    # Set mask to 1 for entries selected by generator
    out_mask[row_indices] = 1
    return out_mask

def sample_from_tensor(inputs, n_samples, with_softmax = False, with_replacement = False):
    if (n_samples >= inputs.size(-1) and not with_replacement) or n_samples < 0 :
        raise ValueError(f"Number of samples({n_samples}) should be positive and lesser than the length of the tensor({inputs.size(-1)}) to sample from.")
    elif n_samples == 0:
        return torch.arange(inputs.size(-1)), inputs
    
    if with_softmax == False:
        indices = torch.multinomial(inputs, n_samples, replacement=with_replacement)
    else:
        indices = torch.multinomial(torch.nn.Softmax(dim=-1)(inputs), n_samples, replacement=with_replacement)

    index_mask = torch.zeros_like(inputs, dtype=inputs.dtype, device=inputs.device)
    masked_outputs =  torch.scatter(index_mask, -1, indices, inputs)
    # TODO: review for batch size > 1 
    index_mask[indices] = 1.
    return index_mask, masked_outputs

def place_batches(batch, devices = []):
    return [{key: value[0].to(device) for key, value in batch.items()} for device in devices]

def evaluate_model(
    model_args,
    training_args,
    generator,
    discriminator,
    ans_discriminator,
    eval_dataloader,
    ):

    # Prepare models for evaluation
    generator.eval()
    discriminator.eval()
    ans_discriminator.eval()

    # Initialize the score lists
    gen_losses, dis_losses, ans_losses = [], [], []
    all_gen_scores, all_dis_scores, all_ans_scores = [], [], []
    all_answerability_labels = []

    # Run through the dataloader
    for batch in tqdm(eval_dataloader, disable=training_args.local_rank  not in {0, -1}):
        [gen_batch, dis_batch, ans_batch] = place_batches(batch, [generator.device, discriminator.device, ans_discriminator.device])
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                """
                gen_scores = generator(**{key: value[0].to(generator.device) for key, value in batch.items()}).logits.squeeze(-1)
                dis_scores = discriminator(**{key: value[0].to(discriminator.device) for key, value in batch.items()}).logits.squeeze(-1)
                ans_scores = ans_discriminator(**{key: value[0].to(ans_discriminator.device) for key, value in batch.items()}).logits.squeeze(-1)
                """
                gen_scores = generator(input_ids=gen_batch["input_ids"],
                            attention_mask=gen_batch["attention_mask"],
                            token_type_ids=gen_batch["token_type_ids"]).logits.squeeze(-1)
                dis_scores = discriminator(input_ids=dis_batch["input_ids"],
                            attention_mask=dis_batch["attention_mask"],
                            token_type_ids=dis_batch["token_type_ids"]).logits.squeeze(-1)
                ans_scores = ans_discriminator(input_ids=ans_batch["input_ids"],
                            attention_mask=ans_batch["attention_mask"],
                            token_type_ids=ans_batch["token_type_ids"]).logits.squeeze(-1)


                # Generator loss
                gen_loss = compute_gen_loss(
                    gen_batch,
                    gen_scores,
                    dis_scores,
                    ans_scores,
                    training_args.n_samples,
                    model_args.ans_discriminator_weight,
                    model_args.regularizer_weight,
                )


                # Begin a discriminator round
                #true_dat_mask = batch["answerability_labels"]
                #true_dis_scores = dis_scores * true_dat_mask.to(dis_scores.device)

                #gen_dat_mask, sample_gen_scores = sample_from_tensor(gen_scores, training_args.n_samples, with_softmax = True, with_replacement = False)
                #gen_dis_scores = dis_scores * gen_dat_mask.to(dis_scores.device)


                # Discriminator loss
                dis_loss = compute_r_dis_loss(
                        dis_batch,
                        dis_scores,
                        gen_scores,
                        training_args.n_samples
                        )
                
                # Answer discriminator loss
                ans_loss = compute_ans_loss(ans_scores, ans_batch)

                gen_losses.append(gen_loss.item())
                dis_losses.append(dis_loss.item())
                ans_losses.append(ans_loss.item())

        all_gen_scores.append(gen_scores.tolist())
        all_dis_scores.append(dis_scores.tolist())
        all_ans_scores.append(ans_scores.tolist())

        all_answerability_labels.append(batch["answerability_labels"].squeeze().tolist())


    print(f"all_gen_scores[0] size = {len(all_gen_scores[0])}")
    print(f"all_dis_scores[0] size = {len(all_dis_scores[0])}")
    print(f"all_ans_scores[0] size = {len(all_ans_scores[0])}")
    print(f"all_answerability_labels[0] size = {len(all_answerability_labels[0])}")


    print(f"all_gen_scores = {all_gen_scores[0][:10]}")
    print(f"all_dis_scores = {all_dis_scores[0][:10]}")
    print(f"all_ans_scores = {all_ans_scores[0][:10]}")
    print(f"all_answerability_labels = {all_answerability_labels[0][:10]}")

        
    #break #TODO: DEBUG only. Remove
    # For each model, calculate its hit score.
    # This is to monitor how each of the models performs in finding ground truth documents
    ret = {}
    ret.update(get_hits_dict(all_gen_scores, all_answerability_labels, model_args.hits_list, "generator"))
    ret.update(get_hits_dict(all_dis_scores, all_answerability_labels, model_args.hits_list, "discriminator"))
    ret.update(get_hits_dict(all_ans_scores, all_answerability_labels, model_args.hits_list, "ans_discriminator"))
    ret["gen_eval_loss"] =  round(sum(gen_losses) / len(gen_losses), 4)
    ret["dis_eval_loss"] =  round(sum(dis_losses) / len(dis_losses), 4)
    ret["ans_eval_loss"] =  round(sum(ans_losses) / len(ans_losses), 4)

    # DEBUG ONLY
    scores = [all_gen_scores, all_dis_scores, all_ans_scores]
    unique_id = uuid.uuid4().hex[0:6]
    for i, model_name in enumerate(["generator", "discriminator", "ans_discriminator"]):
        if  is_abnormal(ret, [model_name], hits = [1, 3, 5, 10, 20, 30, 50]):    
            print(f">>> Saving pickles as {model_name + '_' + unique_id +'.pt'} and {'anslabels'+ '_' + unique_id + '.pt'}")
            with open(model_name+ '_' + unique_id +'.pt', 'wb') as fp:
                pkl.dump(scores[i], fp)
            with open('anslabels'+ '_' + unique_id + '.pt', 'wb') as fp:
                pkl.dump(all_answerability_labels, fp)
    
    return ret

def get_collate_fn(hf_collator):
    return partial(collate_fn, hf_collator=hf_collator)


def collate_fn(features, hf_collator):
    # Input ids are currently in list form
    batch_size = len(features)
    first = features[0]
    num_docs = len(first["input_ids"])
    tokenizer_keys = first.keys()

    # Flatten the batch. ie, convert from list of dict to dict of list
    # and flatten the list so that lists of shape (BxD) become (B*D)
    flattened_batch = {
        key: [doc for datum in features for doc in datum[key]] for key in tokenizer_keys
    }

    # Apply collator and return shape to (BxD)
    padded_flat_batch = hf_collator(flattened_batch)
    reshaped_batch = {
        key: value.reshape(batch_size, num_docs, -1)
        for key, value in padded_flat_batch.items()
    }

    # Convert answerability labels to the correct shape+
    reshaped_batch["answerability_labels"] = reshaped_batch[
        "answerability_labels"
    ].reshape(batch_size, num_docs)
    return reshaped_batch

def remove_dir(dir_path):
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        logger.warn(f"Error: {dir_path} : {e.strerror}")


def calculate_gen_dis_steps(max_steps, num_dis_rounds, num_gen_rounds):
    if max_steps <= 0 or  num_dis_rounds <= 0 or  num_gen_rounds <= 0:
        return 0, 0
    d, g = 0, 0
    rounds, cumulated_rounds = 0, num_dis_rounds + num_gen_rounds
    for _ in range(max_steps):
        if rounds < num_dis_rounds:
            d += 1
            rounds += 1
        elif rounds < cumulated_rounds:
            g += 1
            rounds += 1
        else:
            d += 1
            rounds = 1
    return d, g

def compute_device_map(devices = [0, 1, 2], n_models = 3):
    size = len(devices)
    if size <= 0 or size % n_models != 0:
        raise ValueError(f"Invalid list of devices: should be not empty (size = {size}) and divisible by \'n_models\' ({n_models}) ")
    device_map = {"generator":[], "discriminator":[], "ans_discriminator":[]}
    i = 0
    while i < size:
        device_map["generator"].append(devices[i])
        i += 1
        device_map["discriminator"].append(devices[i])
        i += 1
        device_map["ans_discriminator"].append(devices[i])
        i += 1
    return device_map


def clip_grad_norm(max_grad_norm, scaler, models, optimizers):
    if max_grad_norm <= 0.:
        return

    for model, optimizer in zip(models, optimizers):
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

def step(scaler, optimizers, schedulers):
    for optimizer, scheduler in zip(optimizers, schedulers):
        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)
        scheduler.step()
        #optimizer.zero_grad()

        # Updates the scale for next iteration.
        scaler.update()

def evaluate_and_log(
    model_args, 
    training_args, 
    completed_steps, 
    generator,
    discriminator,
    ans_discriminator,
    eval_dataset,
    data_collator,
    logger,  
    writers = {}, 
    model_name = ""
    ):    
    if completed_steps % training_args.eval_steps != 0:
        return
    
    # Evaluation
    if not model_name:
        model_name = "global_step"
    logger.info(f"#### Evaluating at {model_name} step {completed_steps} ...")

    eval_dataloader = DataLoader(
        IterableDatasetShard(eval_dataset) if isinstance(eval_dataset, torch.utils.data.IterableDataset) else eval_dataset,
        collate_fn=data_collator,
        batch_size=training_args.per_device_eval_batch_size,
        num_workers=int(training_args.dataloader_num_workers)
    )

    """ Evaluate models """
    eval_metrics = evaluate_model(
        model_args, 
        training_args,
        generator,
        discriminator,
        ans_discriminator,
        eval_dataloader
        )
    print_results(eval_metrics)

    if training_args.local_rank in {0, -1}:
        # Log the training steps
        logger.info(f">> Generator eval loss == {eval_metrics['gen_eval_loss']}")
        logger.info(f">> Discriminator eval loss == {eval_metrics['dis_eval_loss']}")
        logger.info(f">> Answerability discriminator eval loss == {eval_metrics['ans_eval_loss']}")
    
    # Logging 
    if completed_steps % training_args.logging_steps == 0:
        for key, value in eval_metrics.items():
            if key.startswith("gen"):
                writers["gen"].add_scalar(key, value, completed_steps)
            elif key.startswith("dis"):
                writers["dis"].add_scalar(key, value, completed_steps)
            else:
                writers["ans"].add_scalar(key, value, completed_steps)

def label_datasets(training_args, raw_datasets, logger, overwrite_cache = False):
    # Telling other threads to wait
    if training_args.local_rank > 0:
        logger.info('Waiting for main process to perform the labeling')
        torch.distributed.barrier()

    # Answerability labeling for each document
    def label_answerability(example):
        def find_whole_answer(w):
            return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search

        compile_func = find_whole_answer(
            "|".join([re.escape(answer) for answer in example["answer"]])
        )
        answerability_labels = list(
            map(bool, (map(compile_func, example["documents"])))
        )
        return {"answerability_labels": answerability_labels}

    logger.info(f"===>>> Labeling the dataset...")
    
    # If test set doesn't contain answers, ignore it
    if "answer" not in raw_datasets["test"].column_names:
        raw_datasets = DatasetDict(
            {"train": raw_datasets["train"], "dev": raw_datasets["dev"]}
        )
    answerable_datasets = raw_datasets.map(
        label_answerability,
        num_proc=training_args.num_processes,
        load_from_cache_file=(not overwrite_cache) or (training_args.local_rank > 0),
        desc="Labeling each document as having an answer or not",
    )

    # Signaling to everyone else that labeling is done
    if training_args.local_rank == 0:
        torch.distributed.barrier()

    # Keep only questions that have at least one answerable document
    answerable_datasets = answerable_datasets.filter(
        lambda example: any(example["answerability_labels"]),
        num_proc=training_args.num_processes,
    )
    return answerable_datasets

def tokenize_datasets(
    data_args,
    training_args, 
    answerable_datasets, 
    column_names,  
    tokenizer, 
    logger, 
    overwrite_cache = False
    ):
   # Tokenize the questions and documents
    # OUSSAMA's
    def tokenize_function(examples):
        # Prepend each document with the question
        list_of_dicts = [
            tokenizer(
                examples["question"],
                document,
                max_length=data_args.max_seq_length,
                truncation=True,
            )
            for document in examples["documents"]
        ]
        first = list_of_dicts[0]
        return {key: [elem[key] for elem in list_of_dicts] for key in first.keys()}


    if training_args.local_rank > 0:
        logger.info('Waiting for main process to perform the tokenization')
        torch.distributed.barrier()

    # Preprocessing the datasets.
    tokenized_datasets = answerable_datasets.map(
        tokenize_function,
        num_proc=training_args.num_processes,
        remove_columns=column_names,
        load_from_cache_file=(not overwrite_cache) or (training_args.local_rank > 0),
        desc="Running tokenizer on every text in dataset",
    )
    # Signaling to everyone else that tokenization is done
    if training_args.local_rank == 0:
        torch.distributed.barrier()
    return tokenized_datasets


""" OLD Implementation
def tokenize_function(examples):
    # Prepend each document with the question
    return tokenizer(
        [
            examples["question"] + tokenizer.sep_token + document
            for document in examples["documents"]
        ],
        #examples["question"],
        #document,
        truncation=True,
        max_length=data_args.max_seq_length,
    )
"""
def init_optimizer(train_args, train_steps, model):
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
            "weight_decay": train_args.weight_decay,
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
    if train_args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(grouped_parameters, lr=train_args.learning_rate)
    elif train_args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(grouped_parameters, lr=train_args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer (\'{train_args.optimizer}\'). Supported optimizers are \'SGD\' and \'AdamW\'")


    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        name=train_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=train_args.warmup_steps,
        num_training_steps=train_steps,
    )
    return optimizer, lr_scheduler

def checkpoint(training_args, completed_steps, model, epoch, logger, force = False):
    # Helper function for saving models
    def save_model(model, output_dir, local_rank):
        unwrap_model(model).save_pretrained(
            output_dir,
            is_main_process=(local_rank in {0, -1}),
        )
    chekpointing_steps = training_args.save_steps

    # Save the model
    if force == False and chekpointing_steps > 0 and completed_steps % chekpointing_steps == 0:
        output_dir_suffix = f"model_epoch_{epoch}_chkpt_step_{completed_steps}"
        # TODO : Getting rid of extra model checkpoints, if need be
        if training_args.local_rank in {-1, 0}:
            output_dir = os.path.join(training_args.output_dir, output_dir_suffix)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"#### Checkpoint at step {completed_steps}: Saving the model at {output_dir}")
            save_model(model, output_dir, training_args.local_rank)
    # Save the final model
    elif force == True and training_args.local_rank in {-1, 0}:
        output_dir_suffix = f"model_epoch_{epoch}_chkpt_step_{completed_steps}"
        output_dir = os.path.join(training_args.output_dir, output_dir_suffix)
        logger.info(f"#### Checkpoint at step {completed_steps}: Saving the model at {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        save_model(model, output_dir, training_args.local_rank)

def print_results(dic):
    logger.info(f"gen_eval_loss = {dic['gen_eval_loss']}")
    logger.info(f"dis_eval_loss = {dic['dis_eval_loss']}")
    logger.info(f"ans_eval_loss = {dic['ans_eval_loss']}")

    logger.info(f"generator_hits@1 = {dic['generator_hits@1']}")
    logger.info(f"generator_hits@3 = {dic['generator_hits@3']}")
    logger.info(f"generator_hits@5 = {dic['generator_hits@5']}")
    logger.info(f"generator_hits@10 = {dic['generator_hits@10']}")
    logger.info(f"generator_hits@20 = {dic['generator_hits@20']}")
    logger.info(f"generator_hits@30 = {dic['generator_hits@30']}")
    logger.info(f"generator_hits@50 = {dic['generator_hits@50']}")

    logger.info(f"discriminator_hits@1 = {dic['discriminator_hits@1']}")
    logger.info(f"discriminator_hits@3 = {dic['discriminator_hits@3']}")
    logger.info(f"discriminator_hits@5 = {dic['discriminator_hits@5']}")
    logger.info(f"discriminator_hits@10 = {dic['discriminator_hits@10']}")
    logger.info(f"discriminator_hits@20 = {dic['discriminator_hits@20']}")
    logger.info(f"discriminator_hits@30 = {dic['discriminator_hits@30']}")
    logger.info(f"discriminator_hits@50 = {dic['discriminator_hits@50']}")

    logger.info(f"ans_discriminator_hits@1 = {dic['ans_discriminator_hits@1']}")
    logger.info(f"ans_discriminator_hits@3 = {dic['ans_discriminator_hits@3']}")
    logger.info(f"ans_discriminator_hits@5 = {dic['ans_discriminator_hits@5']}")
    logger.info(f"ans_discriminator_hits@10 = {dic['ans_discriminator_hits@10']}")
    logger.info(f"ans_discriminator_hits@20 = {dic['ans_discriminator_hits@20']}")
    logger.info(f"ans_discriminator_hits@30 = {dic['ans_discriminator_hits@30']}")
    logger.info(f"ans_discriminator_hits@50 = {dic['ans_discriminator_hits@50']}")



