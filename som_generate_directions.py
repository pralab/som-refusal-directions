from collections import defaultdict
import os
import torch
import argparse
import numpy as np
from config import Config
import warnings
from som import train_som
from dataset.utils import compute_centroid

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--som_x', default=2, type=int, help="number of som neurons in the x-axis")
    parser.add_argument('--som_y', default=2, type=int, help="number of som neurons in the y-axis")
    parser.add_argument('--iterations', default=10000, type=int, help="number of som training iterations")
    parser.add_argument('--lr', default=0.01, type=float, help="som learning rate")
    parser.add_argument('--sigma', default=0.1, type=float, help="value of the radius of each som neuron") 
    parser.add_argument('--layer', default=30, type=int, help="defines the model layer from which extract the embedding") 
    parser.add_argument('--model_name', default="llama2-7b", type=str, help="llm")
    parser.add_argument('--save_weights', default=False, action="store_true", help="if true, save neurons not directions")


    return parser.parse_args()

def load_data(model_name):
    """
    Load the hidden layer representations.
    Adjust the file paths if needed.
    HL_x and HF_x should have shape [n_samples, n_layers, hidden_dimension].
    """

    HL_x = torch.load(f"./dataset/representations/{model_name}/HLx_train.pt", weights_only=True).numpy()
    HF_x = torch.load(f"./dataset/representations/{model_name}/HFx_train.pt", weights_only=True).numpy() 

    # Create labels: 0 for HL, 1 for HF
    Yhl = np.zeros(len(HL_x))
    Yhf = np.ones(len(HF_x))

    print(f"Loaded: \n\nHL_x shape: {HL_x.shape}, HF_x shape: {HF_x.shape}, Yhl shape: {Yhl.shape}, Yhf shape: {Yhf.shape}")
    
    return HL_x, HF_x, Yhl, Yhf


def som_generate_save_dirs(cfg, som, X_in, Y, aux_name, save_weights, centroid=None):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))
    
    if save_weights:
        dir_path = os.path.join(cfg.artifact_path(), f'generate_directions/{aux_name}_weights.pt')
        if os.path.isfile(dir_path):
            print('Reusing already computed directions...')
            return torch.load(dir_path), dir_path
        
        print('Computing Neurons...')
        # this returns a numpy.ndarray som_x X som_y X repr_size
        weights = som.get_weights()
        winners = np.array([som.winner(xi) for xi in X_in]) 
        torch_w = torch.tensor(weights).view(weights.shape[0]*weights.shape[1], weights.shape[2])
        torch.save(torch_w, os.path.join(cfg.artifact_path(), f'generate_directions/{aux_name}_weights.pt') )

        print('[✓] Saved weights at:', os.path.join(cfg.artifact_path(), f'generate_directions/{aux_name}_weights.pt'))
        #neuron_to_prompt_ids = defaultdict(list)
        #for idx, pair in enumerate(winners):
        #    key = str(list(pair))  # Turn the numpy array into a list and stringify
        #    neuron_to_prompt_ids[key].append(idx)      

        return torch.tensor(weights).squeeze(1), os.path.join(cfg.artifact_path(), f'generate_directions/{aux_name}_weights.pt')
    
    else: #save directions 

        dir_path = os.path.join(cfg.artifact_path(), f'generate_directions/{aux_name}_directions.pt')
        print('Computing directions...')
        # this returns a numpy.ndarray som_x X som_y X repr_size
        weights = som.get_weights()
        winners = np.array([som.winner(xi) for xi in X_in]) 

        counts = defaultdict(lambda: [0, 0])  # [class_0_count, class_1_count]
        for winner, label in zip(winners, Y):
            winner_tuple = tuple(winner)  # convert array([i, j]) → (i, j)
            label_int = int(label)        # ensure 0 or 1 as Python int
            counts[winner_tuple][label_int] += 1 
        
        # sort neurons by number of wins for each class
        sorted_class_1 = sorted(counts.items(), key=lambda item: item[1][1], reverse=True)
        #for l in sorted_class_1:
        #    print(l, "\n")
        
        # take top-k neurons for each class
        top_k_class_1 = [item[0] for item in sorted_class_1[:]]
        # Get corresponding weights
        weights_class_1 = np.array([weights[i, j] for (i, j) in top_k_class_1])
        
        w1 = torch.tensor(weights_class_1, dtype=torch.float32)  # shape (k, d)
        directions = torch.stack([(w1_i - centroid) for w1_i in w1]) 
        
        # raise a warning if any direction is all zeros
        for idx, direction in enumerate(directions):
            if torch.all(direction == 0):
                warnings.warn(f"Please refrain from using direction at index {idx}, which is a zero vector.")
                
        # save the tensor
        torch.save(directions, dir_path)
        print(f"[✓] Saved {directions.shape[0]} directions to {dir_path}")

        return directions, dir_path

if __name__ == "__main__":

    # parse arguments 
    args = get_args()
    model_path = args.model_name
    
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
    
    # this will simply load the already generated representation
    HL_x, HF_x, Yhl, Yhf = load_data(args.model_name) 

    X = np.concatenate([HF_x])
    Y = np.concatenate([Yhf])
    X_in = X[:, args.layer, :] 
    som = train_som(X_in, som_x=args.som_x, som_y=args.som_y, iterations=args.iterations, sigma=args.sigma, learning_rate=args.lr)
    c0 = compute_centroid(HL_x, layer=args.layer)
    aux_name = f'centroid_to_som{args.som_x}_sigma{args.sigma}_layer{args.layer}'
    directions, dir_path = som_generate_save_dirs(cfg, som, X_in, Y, aux_name=aux_name, save_weights=args.save_weights, centroid=c0)

