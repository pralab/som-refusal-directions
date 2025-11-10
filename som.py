from collections import defaultdict
import os
import json
import torch
import random
import argparse
import numpy as np 
from minisom import MiniSom
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dataset.utils import compute_centroid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib import cm
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--som_x', default=3, type=int, help="number of som neurons in the x-axis")
    parser.add_argument('--som_y', default=3, type=int, help="number of som neurons in the y-axis")
    parser.add_argument('--iterations', default=10000, type=int, help="number of som training iterations")
    parser.add_argument('--lr', default=0.01, type=float, help="som learning rate")
    parser.add_argument('--sigma', default=0.33, type=float, help="value of the radius of each som neuron")
    parser.add_argument('--pca', default=False, action="store_true", help="runs pca on embedding when true")
    parser.add_argument('--components', default=10, type=int, help="number of PCA components")
    parser.add_argument('--ranked', default=False, action="store_true", help="plots the rank of the neurons on top of the som")
    parser.add_argument('--ol_centr', default=False, action="store_true", help="if true, overlays the centroids on the sam grid plot")
    parser.add_argument('--ol_data', default=False, action="store_true", help="if true, overlays the embedding points on the sam grid plot")
    parser.add_argument('--layer', default=30, type=int, help="defines the model layer from which extract the embedding")
    parser.add_argument('--multiplot', default="", type=str, choices=("layer", "sigma"))
    parser.add_argument('--multiplot_list', nargs='+', type=int, default=[1,2,3])
    parser.add_argument('--find_layer', action='store_true', help="instead of plotting, find which layer's SOM best separates harmful vs. harmless")
    parser.add_argument('--top_k', type=int, default=2, help="when scoring a layer, look only at its top_k neurons by support") 
    parser.add_argument('--model_name', type=str, default="llama3-8b", help="name of the model to use for the representation")
    parser.add_argument('--ranked_plot', action='store_true', help="simply plot the umatrix")
    return parser.parse_args()

def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
def set_filename(args):
    base = "som"

    # add layer info
    if args.multiplot == "":
        base += f"_layer{args.layer}"
    elif args.multiplot == "layer":
        layers_str = "-".join(map(str, args.multiplot_list))
        base += f"_layers_{layers_str}"
    elif args.multiplot == "sigma":
        sigmas_str = "-".join(map(str, args.multiplot_list))
        base += f"_sigmas_{sigmas_str}"

    # add PCA info
    if args.pca:
        base += f"_pca{args.components}"
    else:
        base += "_raw"

    # add SOM grid info
    base += f"_{args.som_x}x{args.som_y}"

    # add learning parameters
    base += f"_lr{args.lr}_sigma{args.sigma}"

    # overlay info
    overlays = []
    if args.ol_data:
        overlays.append("data")
    if args.ol_centr:
        overlays.append("centr")
    if overlays:
        base += "_ol-" + "-".join(overlays)

    # add iterations info
    base += f"_it{args.iterations}"

    # complete filename
    return base + ".pdf"

def plot_ranked_som_neurons_by_harmful_activity_hex(som, X_in, Y, filename="ranked_hexsom.pdf"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from collections import defaultdict
    # just once, at top of your function
    palette = sns.color_palette("Paired")
    paired_green = palette[3]   # this is “#33a02c”

    # make a custom white→green colormap
    custom_green_cmap = LinearSegmentedColormap.from_list(
        "custom_green", ["white", paired_green]
    )


    plt.rcParams.update({
        "font.family": "serif",
        "figure.dpi": 300,
        "savefig.dpi": 300
    })
    # 1) Count harmful samples per neuron
    winners = np.array([som.winner(xi) for xi in X_in])
    harmful_counts = defaultdict(int)
    for w, lbl in zip(winners, Y):
        if int(lbl) == 1:
            harmful_counts[tuple(w)] += 1

    # 2) Fetch weights & U-matrix
    weights = som.get_weights()
    umatrix = som.distance_map()

    # 3) Rank neurons
    coords = [(i,j) for i in range(weights.shape[0])
                    for j in range(weights.shape[1])]
    sorted_neurons = sorted(coords,
                            key=lambda c: harmful_counts.get(c,0),
                            reverse=True)
    neuron_ranks = {c:r for r,c in enumerate(sorted_neurons)}

    # 4) Compute flat-top hex centers
    radius = 1.0
    dx = np.sqrt(3) * radius   # horizontal step
    dy = 1.5       * radius    # vertical step
    centers = {}
    n_rows, n_cols = weights.shape[:2]
    for i in range(n_rows):
        for j in range(n_cols):
            x = j * dx + (i % 2) * (dx / 2.0)
            y = i * dy
            centers[(i,j)] = (x, y)

    # 5) Plot
    fig, ax = plt.subplots(figsize=(11,11))
    ax.set_aspect('equal')

    for (i,j),(x,y) in centers.items():
         # remap into [start_frac → 1], so it never goes all the way to white
        start_frac = 0.1    # 0 = white, 1 = full paired_green; raise to start darker
        u_norm = umatrix[i,j] / umatrix.max()
        shade = start_frac + (1.0 - start_frac) * u_norm
        color = custom_green_cmap(shade)
        hexagon = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=radius,
            orientation=0,   # flat-top
            facecolor=color,
            edgecolor='black', 
            linewidth=2
        )
        ax.add_patch(hexagon)

        r = neuron_ranks.get((i,j))
        if r is not None:
            ax.text(x, y, str(r),
                    ha='center', va='center',
                    fontsize=15, color='black', weight='bold')

    # 6) Force square data window
    xs = np.array([c[0] for c in centers.values()])
    ys = np.array([c[1] for c in centers.values()])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    pad = radius * 1.2

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * max_range + pad

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    ax.set_xticks([]); ax.set_yticks([])

    # 7) Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    sm = cm.ScalarMappable(
    cmap=custom_green_cmap,
    norm=Normalize(vmin=umatrix.min(), vmax=umatrix.max())
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("U-Matrix Distance", rotation=270, labelpad=15, fontsize=15)

    ax.set_title("Hexagonal SOM Neurons", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"logs/{filename}", bbox_inches='tight')
    print(f"[✓] Saved ranked hex SOM to logs/{filename}")
    plt.close()



def load_data(reduce=None, model_name="llama3-8b"):
    """
    Load the hidden layer representations.
    Adjust the file paths if needed.
    HL_x and HF_x should have shape [n_samples, n_layers, hidden_dimension].
    """
    HL_x = torch.load(f"./dataset/representations/{model_name}/mix/HLx_train.pt", weights_only=True).numpy()
    HF_x = torch.load(f"./dataset/representations/{model_name}/mix/HFx_train.pt", weights_only=True).numpy()

    if reduce is not None:
        HL_x = HL_x[0:reduce]
        HF_x = HF_x[0:reduce]
    # Create labels: 0 for HL, 1 for HF
    Yhl = np.zeros(len(HL_x))
    Yhf = np.ones(len(HF_x))
    
    return HL_x, HF_x, Yhl, Yhf

def train_som(data, som_x=500, som_y=500, iterations=10000, sigma=0.01, learning_rate=0.01):
    """
    Create and train a Self-Organizing Map (SOM) using MiniSom.
    
    Parameters:
      data: np.ndarray of shape [n_samples, input_len]
      som_x: int, number of neurons in the x-direction of the grid
      som_y: int, number of neurons in the y-direction of the grid
      iterations: int, number of training iterations
      sigma: float, neighborhood radius
      learning_rate: float, learning rate
      
    Returns:
      som: trained MiniSom instance
    """
    input_len = data.shape[1]
    som = MiniSom(som_x, som_y, input_len, sigma=sigma, learning_rate=learning_rate, random_seed=0, activation_distance='euclidean', topology='hexagonal')
    som.random_weights_init(data)
    print("Training SOM...")
    som.train_random(data, iterations)
    print("SOM training complete.")
    return som

def plot_som_u_matrix(som, som_x=500, som_y=500, filename='plot.png'):
    """
    Plot the U-Matrix (distance map) of the trained SOM.
    Darker regions indicate larger distances between neurons.
    """
    plt.figure(figsize=(8, 8))
    # Transpose so that the x and y coordinates match the grid layout
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar(label='Distance')
    plt.title("SOM U-Matrix")
    plt.xlabel("SOM x")
    plt.ylabel("SOM y")
    plt.gca().invert_yaxis()  # Invert y-axis to match typical matrix layout
    plt.savefig(filename)

def evaluate_layer_som(
    X_layer: np.ndarray,
    Y: np.ndarray,
    som_x: int, som_y: int,
    iterations: int, sigma: float, learning_rate: float,
    pca: PCA | None,
    top_k: int
) -> tuple[float, list[tuple[int,int]]]:
    """
    Train an SOM on X_layer and return:
      - avg_diff = average of (c0−c1) over top_k class-0 neurons
                 and (c1−c0) over top_k class-1 neurons
      - list of the 2*top_k counts: [(c0,c1),...] in that order
    """
    # optionally PCA
    if pca is not None:
        X_in = pca.fit_transform(X_layer)
    else:
        X_in = X_layer

    # train
    som = train_som(
        X_in,
        som_x=som_x, som_y=som_y,
        iterations=iterations,
        sigma=sigma,
        learning_rate=learning_rate
    )

    # count hits per neuron
    winners = np.array([som.winner(xi) for xi in X_in])
    counts = defaultdict(lambda: [0, 0])
    for (wx, wy), label in zip(winners, Y):
        counts[(wx, wy)][int(label)] += 1

    # top_k for class 0 by c0 desc
    class0 = sorted(counts.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
    # top_k for class 1 by c1 desc
    class1 = sorted(counts.items(), key=lambda kv: kv[1][1], reverse=True)[:top_k]

    diffs = []
    combined = []
    for _, (c0, c1) in class0:
        diffs.append(c0 - c1)
        combined.append((c0, c1))
    #for _, (c0, c1) in class1:
    #    diffs.append(c1 - c0)
    #    combined.append((c0, c1))

    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    return avg_diff, combined


def plot_som_winners(som, data, labels, som_x=500, som_y=500, filename='plot.png'):
    """
    Overlay the winning neuron for each sample on the SOM grid.
    Each sample's winning neuron is marked with its class label.
    """
    plt.figure(figsize=(som_x, som_y))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar(label='Distance')
    plt.title("SOM Winners with Class Labels")
    
    for idx, sample in enumerate(data):
        winner = som.winner(sample)
        # Offset the text to center in the cell; use red for class 1 and blue for class 0
        color = 'red' if labels[idx] == 1 else 'blue'
        plt.text(winner[0] + 0.5, winner[1] + 0.5, str(int(labels[idx])),
                 color=color, fontdict={'weight': 'bold', 'size': 9},
                 ha='center', va='center')
    plt.xlabel("SOM x")
    plt.ylabel("SOM y")
    plt.gca().invert_yaxis()
    plt.savefig(filename)

def plot_som_ranked_neurons(
    som,
    data: np.ndarray,
    labels: np.ndarray,
    som_x: int = 30,
    som_y: int = 30,
    filename: str = 'ranked_neurons.png'
):
    """
    Plot top-k most winning neurons for harmful (label=1) and harmless (label=0) samples.
    Labels are shown as: 0HF, 1HF, ..., 0HL, 1HL, ...
    """
    winners = np.array([som.winner(xi) for xi in data]) 
    counts = defaultdict(lambda: [0, 0])  # [class_0_count, class_1_count]

    for winner, label in zip(winners, labels):
        winner_tuple = tuple(winner)  # convert array([i, j]) → (i, j)
        label_int = int(label)        # ensure 0 or 1 as Python int
        counts[winner_tuple][label_int] += 1 
    
    # sort neurons by number of wins for each class
    sorted_class_0 = sorted(counts.items(), key=lambda item: item[1][0], reverse=True)
    sorted_class_1 = sorted(counts.items(), key=lambda item: item[1][1], reverse=True)

    # take top-k neurons for each class
    top_k_class_0 = [item[0] for item in sorted_class_0[:top_k]]
    top_k_class_1 = [item[0] for item in sorted_class_1[:top_k]]
    print(top_k_class_0, top_k_class_1)
    # Sort neurons separately by HL and HF frequency
    sorted_HL = sorted(win_counts.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
    sorted_HF = sorted(win_counts.items(), key=lambda x: x[1][1], reverse=True)[:top_k]

    # Extract the neuron positions
    hl_winners = [pos for (pos, _) in sorted_HL]
    hf_winners = [pos for (pos, _) in sorted_HF]

    # Plot U-matrix as background
    plt.figure(figsize=(10, 10))
    plt.pcolor(som.distance_map().T, cmap='bone_r')  # U-Matrix
    plt.colorbar(label='Distance')
    plt.title("Top-K Ranked Neurons by Class")
    plt.gca().invert_yaxis()

    # Plot HL winners
    for idx, (x, y) in enumerate(hl_winners):
        plt.text(x + 0.5, y + 0.5, f"{idx}HL", color='blue', fontsize=12, weight='bold',
                 ha='center', va='center', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))

    # Plot HF winners
    for idx, (x, y) in enumerate(hf_winners):
        plt.text(x + 0.5, y + 0.5, f"{idx}HF", color='red', fontsize=12, weight='bold',
                 ha='center', va='center', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

    plt.xlabel("SOM x")
    plt.ylabel("SOM y")
    plt.savefig(filename)
    print(f"[✓] Top-k neurons plot saved to {filename}")

def plot_umatrices_over_sigma(X, pca, pca_components=10, som_x=30, som_y=30, iterations=3000, sigmas_list=[1,2,3], filename='plot.png'):
    """
    Plot U-Matrices of SOMs trained on PCA-reduced data with varying numbers of components.
    """
    n_cols = 5
    n_rows = int(np.ceil(len(sigmas_list) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    for idx, n_comp in enumerate(sigmas_list):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]

        X_in = X
        if pca:
            pca = PCA(n_components=pca_components)
            X_in = pca.fit_transform(X)

        som = train_som(X_in, som_x=som_x, som_y=som_y, iterations=iterations, learning_rate=0.01, sigma=n_comp)

        u_matrix = som.distance_map().T
        im = ax.imshow(u_matrix, cmap='bone_r', origin='lower')
        ax.set_title(f"Sigma: {n_comp}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any empty subplots
    for idx in range(len(sigmas_list), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')

    fig.suptitle("SOM U-Matrix over Sigma Values", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename)

def plot_ranked_som_neurons_by_harmful_activity(som, X_in, Y, som_x, som_y, filename="ranked_som_neurons.png"):
    """
    Plot the U-Matrix of the SOM and annotate each neuron with its rank based on how many harmful (label=1) points it captures.
    """

    # Count how many harmful points (label=1) each neuron wins
    winners = np.array([som.winner(xi) for xi in X_in])
    harmful_counts = defaultdict(int)
    for winner, label in zip(winners, Y):
        if int(label) == 1:
            harmful_counts[tuple(winner)] += 1

    # Sort all neurons by harmful count (descending)
    all_coords = [(i, j) for i in range(som_x) for j in range(som_y)]
    sorted_neurons = sorted(all_coords, key=lambda k: harmful_counts.get(k, 0), reverse=True)

    # Assign ranks: rank 0 is the neuron with most harmful samples
    neuron_ranks = {coord: rank for rank, coord in enumerate(sorted_neurons)}

    # Plot U-Matrix
    plt.figure(figsize=(10, 10))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar(label='Distance')
    plt.gca().invert_yaxis()
    plt.title("SOM Neuron Ranks (Harmful-Focused)")

    # Overlay ranks
    for (x, y), rank in neuron_ranks.items():
        plt.text(x + 0.5, y + 0.5, str(rank), ha='center', va='center',
                 fontsize=8, color='black', weight='bold',
                 bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'))

    plt.xlabel("SOM x")
    plt.ylabel("SOM y")
    plt.savefig(f'logs/{filename}.pdf', bbox_inches='tight')
    print(f"[✓] SOM neuron ranks saved to {filename}")


def plot_umatrices_over_layers(X, pca=False, pca_components=10, layers=[1,2,3], som_x=30, som_y=30, iterations=3000, sigma=25, filename='plot.png'):
    """
    Plot U-Matrices of SOMs trained on PCA-reduced data with varying numbers of components.
    """
    n_cols = 4
    n_rows = int(np.ceil(len(layers) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    for idx, layer in enumerate(layers):

        row, col = divmod(idx, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        X_layer = X[:, layer, :] 
        X_in = X_layer
        if pca:
            pca = PCA(n_components=pca_components)
            X_in = pca.fit_transform(X_layer)

        som = train_som(X_in, som_x=som_x, som_y=som_y, iterations=iterations, learning_rate=0.01, sigma=sigma)

        u_matrix = som.distance_map().T
        im = ax.imshow(u_matrix, cmap='bone_r', origin='lower')
        ax.set_title(f"Layer: {layer}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any empty subplots
    for idx in range(len(layers), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')

    fig.suptitle("SOM U-Matrix over Layers", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename)


def plot_overlay_data_on_som(som, pca, X_layer, Y, c0, c1, midp, hidden_dim, layer, ol_centr=True, ol_data=True, filename='plot.png'):
    """
    Show the full SOM grid with class-labeled data points, centroids, and midp overlaid.
    """
    # data
    c0_in = c0 
    c1_in = c1 
    midp_in = midp 
    
    if pca != None: 
        # Transform centroids and midp to PCA space
        c0_in = pca.transform(c0.reshape(1, hidden_dim))[0]
        c1_in = pca.transform(c1.reshape(1, hidden_dim))[0]
        midp_in = pca.transform(midp.reshape(1, hidden_dim))[0]

    # Find BMUs on the SOM
    c0_bmu = som.winner(c0_in)
    c1_bmu = som.winner(c1_in)
    midp_bmu = som.winner(midp_in)

    # Start plotting
    plt.figure(figsize=(10, 10))
    plt.pcolor(som.distance_map().T, cmap='bone_r')  # U-Matrix
    plt.colorbar(label='Distance')
    plt.title("SOM with Data, Centroids, and midp")
    plt.gca().invert_yaxis()
     
    
    # Plot each data point at its BMU 
    if ol_data:
        for idx, sample in enumerate(X_layer):  
            x = sample
            if pca != None:
                x = pca.transform(sample.reshape(1, -1))[0]
            bmu = som.winner(x)
            color = 'blue' if Y[idx] == 0 else 'red'
            plt.plot(bmu[0]+0.5, bmu[1]+0.5, 'o', color=color, markersize=3, alpha=0.5)

    # Plot centroids 
    if ol_centr:
        plt.plot(c0_bmu[0]+0.5, c0_bmu[1]+0.5, marker='D', color='blue', markersize=10, label='Centroid 0')
        plt.plot(c1_bmu[0]+0.5, c1_bmu[1]+0.5, marker='D', color='red', markersize=10, label='Centroid 1')
        plt.plot(midp_bmu[0]+0.5, midp_bmu[1]+0.5, marker='o', color='purple', markersize=10, label='midp')

    plt.legend(loc='upper right')
    plt.xlabel("SOM x")
    plt.ylabel("SOM y")
    plt.savefig(filename)
    

def main():
    
    # set seed 
    set_seeds()
    
    # parse arguments 
    args = get_args() 
    
    # set filename
    filename = set_filename(args)
    
    # load the split dataset 
    HL_x, HF_x, Yhl, Yhf = load_data(model_name=args.model_name)
    # concatenate data
    X = np.concatenate([HL_x, HF_x])
    Y = np.concatenate([Yhl, Yhf]) 

    if args.ranked_plot:
        HF_layer = HF_x[:, args.layer, :]
        som = train_som(HF_layer, som_x=args.som_x, som_y=args.som_y, iterations=args.iterations, learning_rate=args.lr, sigma=args.sigma)
        plot_ranked_som_neurons_by_harmful_activity_hex(som=som,
                                X_in=HF_layer,       
                                Y=Yhf,
                                filename="ranked"+filename) 
        return

    
    if args.find_layer:
        # load all layers
        HL_x, HF_x, Yhl, Yhf = load_data()
        X = np.concatenate([HL_x, HF_x], axis=0)
        Y = np.concatenate([Yhl, Yhf], axis=0)
        pca_obj = PCA(n_components=args.components) if args.pca else None

        best_layer, best_score, best_counts = -1, float('-inf'), []
        for layer in range(X.shape[1]):
            score, counts = evaluate_layer_som(
                X[:, layer, :], Y,
                som_x=args.som_x,
                som_y=args.som_y,
                iterations=args.iterations,
                sigma=args.sigma,
                learning_rate=args.lr,
                pca=pca_obj,
                top_k=args.top_k
            )
            print(
                f"Layer {layer:2d} → avg_diff {score:.4f}, "
                f"top_2*{args.top_k} counts: {counts}"
            )
            if score > best_score:
                best_score, best_layer, best_counts = score, layer, counts

        print(
            f"\n>> Best layer: {best_layer} "
            f"(avg_diff {best_score:.4f}, counts {best_counts})"
        )
        return

    # single plot 
    if args.multiplot == "": 
        # define main parameters 
        layer = args.layer  
        X_layer = X[:, layer, :]  # shape: [n_samples, hidden_dimension]

        # process data if PCA, raw hidden state otherwise 
        if args.pca: 
            pca = PCA(n_components=args.components)
            X_in = pca.fit_transform(X_layer)
        else: 
            X_in = X_layer
            pca = None
            
        # train som 
        som = train_som(X_in, som_x=args.som_x, som_y=args.som_y, iterations=args.iterations, learning_rate=args.lr, sigma=args.sigma)

        # overlay data 
        if args.ol_centr or args.ol_data: 
            difference, midp, mu, v = compute_mean_difference(HF_x, HL_x, layer)
            c0 = compute_centroid(HL_x, layer)
            c1 = compute_centroid(HF_x, layer)
            # plot overlayed data
            plot_overlay_data_on_som(som, pca, 
                                X_in, Y, 
                                c0, c1, midp, 
                                4096, 
                                layer, ol_centr=args.ol_centr, 
                                ol_data=args.ol_data, filename=filename)

        elif not args.ranked: 
            plot_som_u_matrix(som, 
                              som_x=args.som_x, 
                              som_y=args.som_y, 
                              filename=filename)
        
        elif args.ranked: 
            plot_som_ranked_neurons(som=som,
                                    data=X_in,       
                                    labels=Y,
                                    top_k=4,
                                    som_x=args.som_x,
                                    som_y=args.som_y,
                                    filename="ranked"+filename
)

    # do plot over different layers or sigma values
    elif args.multiplot == 'layer': 
        plot_umatrices_over_layers(X, args.pca, 
                                   pca_components=args.components, 
                                   layers=args.multiplot_list, 
                                   som_x=args.som_x, 
                                   som_y=args.som_y, 
                                   iterations=args.iterations, 
                                   sigma=args.sigma, 
                                   filename=filename)
        
    elif args.multiplot == 'sigma': 
        plot_umatrices_over_sigma(X, args.pca, 
                                  pca_components=args.components, 
                                  layers=args.layer, 
                                  som_x=args.som_x, 
                                  som_y=args.som_y, 
                                  iterations=args.iterations, 
                                  sigma=args.multiplot_list, 
                                  filename=filename)
    

if __name__ == "__main__":
    main()
