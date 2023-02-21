from functools import partial
import numpy as np
import torch
from sklearn.metrics import ndcg_score


def get_index_list(batch_field, indices):
    return torch.stack(
        [batch_field[i][j] for i, index_list in enumerate(indices) for j in index_list]
    )


def generate_dis_samples(batch, scores):
    num_docs = batch["input_ids"].shape[1]
    mask = batch["answerability_labels"]

    # Normalize scores so that they sum to 1 with no roundoff error
    scores = scores.double()
    scores = torch.divide(scores.T, scores.sum(dim=1)).T
    indices = [
        np.random.choice(
            range(num_docs), int(row_mask.sum()), p=row_scores, replace=False
        )
        for row_mask, row_scores in zip(mask, scores)
    ]
    out_batch = {key: get_index_list(value, indices) for key, value in batch.items()}
    sample_scores = get_index_list(scores, indices)
    return sample_scores, out_batch


def get_generator_sample_mask(batch, scores):
    batch_size = batch["input_ids"].shape[0]
    num_docs = batch["input_ids"].shape[1]
    mask = batch["answerability_labels"]

    # Normalize scores so that they sum to 1 with no roundoff error
    scores = scores.detach().double()
    scores = torch.divide(scores.T, scores.sum(dim=1)).T

    out_mask = torch.zeros(mask.shape, dtype=torch.bool)

    for i in range(batch_size):
        # Sample as many generator-chosen documents as there are answerable documents
        row_indices = np.random.choice(
            range(num_docs), int(mask[i].sum()), p=scores[i], replace=False
        )

        # Set mask to 1 for entries selected by generator
        out_mask[i][row_indices] = 1
    return out_mask


def generate_gen_samples(batch, scores):
    num_docs = batch["input_ids"].shape[1]
    mask = torch.ones(batch["answerability_labels"].shape)
    num_labels = int(mask.sum())

    # Normalize scores so that they sum to 1 with no roundoff error
    scores = scores.double()
    scores = torch.divide(scores.T, scores.sum(dim=1)).T
    indices = [
        np.random.choice(range(num_docs), int(mask.sum()), p=scores, replace=False)
        for mask, scores in zip(mask, scores)
    ]
    out_batch = {key: get_index_list(value, indices) for key, value in batch.items()}
    out_batch["labels"] = torch.zeros(num_labels)
    return out_batch


def compute_gen_loss(
    batch, gen_scores, dis_scores, ans_scores=None, ans_dis_weight=0.25, reg_weight=1
):
    answerability_mask = batch["answerability_labels"]
    # For each batch of questions, Find the uniform probability of picking a ground truth document
    # prob_true = P(d), d ~ p_true, for each question
    prob_true = 1 / torch.sum(answerability_mask, dim=1)  # (B)

    # Log of probability generator ascribes to ground truth documents
    # Assuming d~p_true, compute log(gen_score(d)),
    prob_pred = torch.mul(answerability_mask, torch.log(gen_scores))  # (B x D)

    # Sum of log(gen_score(d)), d~p_true. Sums over all documents.
    prob_pred = torch.sum(prob_pred, dim=1)  # (B)

    # Expected value of log generator score w.r.t. a ground truth document
    # E_{d~p_true}(sum(log(gen_score(d))))
    regularizer = torch.matmul(prob_true, prob_pred)  # (1)

    # Sum of xpected value of discriminator reward
    # E(log(1-sigmoid(discriminator(d))), d~p_gen
    rank_dis_reward = torch.sum(
        torch.mul(gen_scores, torch.log(1 - dis_scores))
    )  # (1)

    if ans_scores is not None:
        ans_dis_reward = torch.sum(
            torch.mul(gen_scores, torch.log(ans_scores))
        )
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
        #answerability_labels = torch.tensor(answerability_mask, dtype=torch.float32)
        answerability_labels = torch.tensor(answerability_mask, dtype=gen_scores.dtype)
        ans_dis_reward = loss_fn(gen_scores, answerability_labels)

    loss = rank_dis_reward - ans_dis_weight * ans_dis_reward - reg_weight * regularizer
    return loss


def compute_r_dis_loss(
    true_dis_scores,
    gen_dis_scores,
    answerability_mask,
    gen_scores,
):
    # Assumes true_dis_scores, gen_dis_scores are of size (L)
    num_docs = answerability_mask.shape[1]

    # Compute probability of answerable question for each row
    prob_true = 1 / torch.sum(answerability_mask, dim=1)  # (B)

    # Repeat probability over all columns and pick positions using mask
    # This is done so that the document probabilities are the same order
    # as in the flattened version
    prob_true_labels = prob_true.unsqueeze(dim=1).repeat(1, num_docs)[
        answerability_mask
    ]  # (L)

    # Expected value of log(sigmoid(discriminator(d))), d~p_true
    reward_true_dat = torch.sum(
        torch.mul(prob_true_labels, torch.log(torch.sigmoid(true_dis_scores)))
    )

    # Expected value of log(1-sigmoid(discriminator(d))), d~p_gen
    reward_gen_dat = torch.sum(
        torch.mul(gen_scores, torch.log(1 - torch.sigmoid(gen_dis_scores)))
    )

    loss = -torch.add(reward_true_dat, reward_gen_dat)
    return loss


def compute_hits(
    all_scores, all_answerability_labels, hits_list=[1, 3, 5, 10, 20, 30, 50]
):
    pred_labels = torch.sigmoid(torch.tensor(all_scores)).tolist()
    ndcg_gen_hits = [
        ndcg_score(all_answerability_labels, pred_labels, k=hits) for hits in hits_list
    ]
    return ndcg_gen_hits


def get_hits_dict(all_scores, all_answerability_labels, hits_list, model_name):
    hits_scores = compute_hits(all_scores, all_answerability_labels, hits_list)
    return {
        model_name + "_hits@" + str(hit): h_score
        for hit, h_score in zip(hits_list, hits_scores)
    }


def get_collate_fn(hf_collator):
    return partial(collate_fn, hf_collator=hf_collator)


def collate_fn(features, hf_collator):
    # Input ids are currently in list form
    batch_size = len(features)
    first = features[0]
    num_docs = len(first["input_ids"])
    tokenizer_keys = first.keys()
    # Only perform flattening on the keys that tokenizer must pad
    flattened_batch = {
        key: [doc for datum in features for doc in datum[key]] for key in tokenizer_keys
    }
    padded_flat_batch = hf_collator(flattened_batch)
    reshaped_batch = {
        key: value.reshape(batch_size, num_docs, -1)
        for key, value in padded_flat_batch.items()
    }
    reshaped_batch["answerability_labels"] = reshaped_batch[
        "answerability_labels"
    ].squeeze()
    return reshaped_batch
