import os
import torch
from models.load_models import load_model
from utils.ablation_utils import ablate_weights
from config import Config
from directions_ablation import generate_and_save_hookfree_completions
import argparse
 
import re
 

def extract_direction_prefix(filepath: str) -> str:
    basename = os.path.basename(filepath)
    match = re.search(r"((?:\w+_)?som[^\s/]+_layer\d{1,2})", basename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract prefix from filename: {filepath}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="llama2-7b", type=str, help="llm")
    parser.add_argument("--directions_path", type=str, required=True, help="Path to the .pt file containing direction vector.")
    parser.add_argument("--dir_ids", nargs='+', type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default="harmbench_test", help="Name of dataset to run on.")
    parser.add_argument('--device', default="cuda:0", type=str)
    args = parser.parse_args()
 
    # model and config
    device = torch.device(args.device)
    model = load_model(args.model_name, device=device)
    model_path = args.model_name
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
 
    directions = torch.load(args.directions_path)
    multi_dirs = directions[args.dir_ids] if args.dir_ids else directions
    aux_name = 'raw_dirs'
    print('Using raw directions')
    
    # Load direction
    len_dir = len(multi_dirs.shape)
    print(f"→ Loaded {len_dir}D direction vectors from {args.directions_path} with shape {multi_dirs.shape}")


    for idx, sing_dir in enumerate(multi_dirs):
        aux_name += '_'+str(args.dir_ids[idx])
        ablate_weights(model, sing_dir)
    
    # completions name
    aux_name = aux_name+'_'+extract_direction_prefix(args.directions_path)
 
    # Run generation
    generate_and_save_hookfree_completions(
        cfg=cfg,
        folder='completions',
        model_base=model,
        dataset_name=args.dataset_name,
        aux_name=aux_name
    )