import torch
from jaxtyping import Float, Int
import argparse

from models.load_models import load_model
from dataset.load_dataset import load_dataset, load_dataset_split
from dataset.utils import compute_centroid
from .ablation_utils import ablate_weights
import random
from config import Config
import os
import json

# adapted from https://github.com/andyrdt/refusal_direction/blob/main/pipeline/submodules/select_direction.py

def kl_div_fn(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    epsilon: Float=1e-6):
    """
    Compute the KL divergence loss between two tensors of logits.
    """
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)


    return torch.mean(kl_divs, dim=-1)

def get_last_position_logits(model, dataset):

    prompts = [model._get_prompt(p['instruction']) for p in dataset]
    tokenized_instructions = model.tokenizer(prompts,
                padding=True,
                truncation=False,
                return_tensors="pt")

    logits = model.model(
        input_ids=tokenized_instructions.input_ids.to(model.device),
        attention_mask=tokenized_instructions.attention_mask.to(model.device),
    ).logits

    last_position_logits = logits[:, -1, :]

    return last_position_logits


def filter_data (model, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    harmful_train_scores = get_refusal_scores(model, harmful_train)
    harmless_train_scores = get_refusal_scores(model, harmless_train)
    harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
    harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)


    harmful_val_scores = get_refusal_scores(model, harmful_val)
    harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
    harmless_val_scores = get_refusal_scores(model, harmless_val)
    harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

    return harmful_train, harmless_train, harmful_val, harmless_val


def load_and_sample_datasets():
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train'), 128)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train'), 128)
    harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val'), 32)
    harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val'), 32)
    
    return harmful_train, harmless_train, harmful_val, harmless_val


def refusal_score(
    logits: torch.Tensor,
    refusal_toks: Int,
    epsilon: Float = 1e-8,
):
    logits = logits.to(torch.float64)

    # we only care about the last tok position
    logits = logits[:, -1, :]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_probs = probs[:, refusal_toks].sum(dim=-1)

    nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)

def get_refusal_scores(model, dataset):
    
    refusal_scores = torch.zeros(len(dataset), device=model.device)
    prompts = [model._get_prompt(p['instruction']) for p in dataset]

    tokenized_instructions = model.tokenizer(prompts,
                padding=True,
                truncation=False,
                return_tensors="pt")
    
    logits = model.model(
            input_ids=tokenized_instructions.input_ids.to(model.device),
            attention_mask=tokenized_instructions.attention_mask.to(model.device),
        ).logits
    
    refusal_scores = refusal_score(logits=logits, refusal_toks=model.refusal_token_id)

    return refusal_scores


def compute_train_dataset(model, data_hl, data_hf):

    HL = torch.zeros(len(data_hl), model.num_layer, model.hidden_dimension)
    HF = torch.zeros(len(data_hf), model.num_layer, model.hidden_dimension)

    for i in range(0, len(data_hl)):
        prompt_HL = data_hl[i]['instruction']

        HL[i] = model.get_representations(prompt=prompt_HL, token_pos=-1)
    for i in range(0, len(data_hf)):
        prompt_HF = data_hf[i]['instruction']

        HF[i] = model.get_representations(prompt=prompt_HF, token_pos=-1)

    return HF, HL

def compute_single_direction(HF, HL, layer):

    sd = compute_centroid(HF, layer) - compute_centroid(HL, layer)

    return sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to evaluate')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to run the model on')
    args = parser.parse_args()

    # Load model
    model = load_model(args.model_name, args.device, system_prompt=None)
    clean_model = load_model(args.model_name, 'cuda:2', system_prompt=None)
    

    model_path = args.model_name
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
    
    prune_layer_percentage=0.2

    # Load dataset
    hf_train, hl_train, hf_val, hl_val = load_and_sample_datasets()

    scores = {}
    layers = int(model.num_layer*(1 - prune_layer_percentage))

    #hf_train, hl_train, hf_val, hl_val = filter_data(model, hf_train, hl_train, hf_val, hl_val) # original version's filtering mechanism
    #print(f'Filtered Datasets Sizes -> Harmful Train: {len(hf_train)}, Harmless Train: {len(hl_train)}, Harmful Val: {len(hf_val)}, Harmless Val: {len(hl_val)}')
    HF, HL = compute_train_dataset(model, hl_train, hf_train)

    baseline_harmless_logits = get_last_position_logits(model=model, dataset=hl_val)

    for layer in range(layers):

        single_dir = compute_single_direction(HF, HL, layer)
        # Ablate model with sd
        ablate_weights(model, single_dir)
        score = get_refusal_scores(model, hf_val).mean().item()
        intervention_logits = get_last_position_logits(model=model, dataset=hl_val)
        kl_div_score = kl_div_fn(baseline_harmless_logits, intervention_logits).mean(dim=0).item()
        scores[layer] = {"Refusal": score, "KL_Div": kl_div_score}
        print(f'Layer {layer}: Refusal Score = {score}, KL Divergence = {kl_div_score}')

        # Reset model weights
        model.model.load_state_dict(clean_model.model.state_dict())

    valid = {k: v for k, v in scores.items() if v["KL_Div"] < 0.1} # previous work's threshold on KL Divergence

    if valid:
        best_layer = min(valid, key=lambda x: valid[x]["Refusal"])
    else:
        print("[WARNING] No layer has KL < 0.1.")
        best_layer = min(scores, key=lambda x: scores[x]["Refusal"])

    print(f'Best Layer Selected: {best_layer} with Refusal Score = {scores[best_layer]["Refusal"]} and KL Divergence = {scores[best_layer]["KL_Div"]}')    

    folder_path = os.path.join(cfg.artifact_path(), "layer_selection")
    base_file_name = f'layer_selection.json'
    file_path = os.path.join(folder_path, base_file_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print('Created folder:', folder_path)

    with open(file_path, "w") as f:
        json.dump(scores, f, indent=4)


if __name__ == "__main__":
    main()