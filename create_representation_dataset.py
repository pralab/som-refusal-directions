import torch
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm
import os
from models.language_models import Llama2_7b, Llama3_8b, Llama2_13b, Qwen_7b, Qwen_14b, Qwen2_3b, Qwen2_7b, Mistral7B_RR, Zephyr_R2D2, Gemma2_9b
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', default=False, action="store_true", help="if true, token position refers to the generated tokens")
    parser.add_argument('--token_pos', default=-1, type=int, help="token position")
    parser.add_argument('--device', default="cuda", type=str, help="cuda device")
    parser.add_argument('--model_name', default="llama2-7b", type=str, help="llm")
    return parser.parse_args()

def login_huggingface():
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token, add_to_git_credential=True)

def load_and_prepare_datasets():

    ds_HF = load_dataset("sorry-bench/sorry-bench-202406")
    ds_HL = load_dataset("tatsu-lab/alpaca")

    ds_HF = ds_HF.filter(lambda x: x['prompt_style'] != 'uncommon_dialects')
    ds_HF = ds_HF.filter(lambda x: x['prompt_style'] != 'translate-ta')
    ds_HF = ds_HF.filter(lambda x: x['prompt_style'] != 'translate-ml')
    ds_HF = ds_HF.filter(lambda x: x['prompt_style'] != 'atbash')
    ds_HF = ds_HF.filter(lambda x: x['prompt_style'] != 'ascii')
    ds_HF = ds_HF.filter(lambda x: x['prompt_style'] != 'morse')
    ds_HF = ds_HF.filter(lambda x: x['prompt_style'] != 'caesar')
    

    sorry_bench = ds_HF['train'].select_columns(["question_id","category","turns"])
    alpaca = ds_HL['train'].select_columns(["instruction"])
    alpaca = alpaca.rename_column("instruction", "turns").map(lambda x: {"turns": x["turns"], "label": 0})

    sorry_bench =  sorry_bench.select(range(4000))
    alpaca = alpaca.select(range(6000))

    sorry_bench = sorry_bench.shuffle(seed=42)
    alpaca = alpaca.shuffle(seed=42)
    
    alpaca = alpaca.filter(lambda x: x["turns"].strip() != "")

    sb = []
    al = []
    for example in sorry_bench:
        sb.append(example["turns"][0])
    for example in alpaca:
        al.append(example["turns"])

    return sb, al
  

def get_model(model_name, device):
    models = {
        'llama2-7b': Llama2_7b,
        'llama3-8b': Llama3_8b,
        'llama2-13b': Llama2_13b,
        'qwen-7b': Qwen_7b,
        'qwen-14b': Qwen_14b,
        'qwen2-3b': Qwen2_3b,
        'qwen2-7b': Qwen2_7b,
        'mistral-7b-rr': Mistral7B_RR,
        'r2d2': Zephyr_R2D2,
        'gemma2-9b': Gemma2_9b
    }
    print(f"[DEBUG] Initializing model: {model_name} on device: {device}")
    return models[model_name](device=device) #, quantization_config="8bit")

def convert(data, model, generate, token_pos, verbose=False):

    batch_size = len(data)
    print(f"Processing {batch_size} samples with token position {token_pos} using model {model.model_name}")
    hidden_states = torch.zeros(batch_size, model.num_layer, model.hidden_dimension)
    for i, prompt in enumerate(tqdm(data, desc="Processing batches")): 

        if verbose:
            print(f"[DEBUG] Processing prompt {i}: {prompt}")
        if generate and prompt is not None: 
            hidden_states[i] = model.get_representations_generate(prompt=prompt, token_pos=token_pos) 
        elif not generate and prompt is not None:
            hidden_states[i] = model.get_representations(prompt=prompt, token_pos=token_pos)
    return hidden_states
    

def convert_samples(
    model_name: str, generate: bool, token_pos: int, device: str):
    """
    Convert prompts to hidden state representations and save them as a dataset.

    Args:
        csv_path (str): Path to the CSV file containing samples.
        text_column (str): Name of the column containing text data.
        label_column (str): Name of the column containing labels.
        model (Llama7b): The model to use for generating hidden states.
        batch_size (int): Batch size for processing. Default is 512.
        save_dir (str): Directory to save the representations dataset.

    """
    print(device)
    model = get_model(model_name, device=device)

    output_dir = f'./dataset/representations/{model_name}/'
    if generate:
         output_dir = f'./dataset/representations/{model_name}/generate_token_pos_{token_pos}'
    os.makedirs(output_dir, exist_ok=True)

    

    sorry_b, alpaca = load_and_prepare_datasets()
    hidden_states = convert(alpaca, model, generate, token_pos, verbose=True)
    torch.save(hidden_states[:, :, :], output_dir + f'/HLx_train.pt')
    hidden_states = convert(sorry_b, model, generate, token_pos, verbose=True)
    torch.save(hidden_states[:, :, :], output_dir + f'/HFx_train.pt')
 
    del model
    torch.cuda.empty_cache()

 
if __name__ == "__main__":

    args = get_args()  
    #login_huggingface()
    
    result = convert_samples(
            model_name=args.model_name,
            generate=args.generate, 
            token_pos=args.token_pos,
            device=args.device,
        )
