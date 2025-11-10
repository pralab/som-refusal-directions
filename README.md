# SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models
Code used for the experiments. 

### Setup 
First, login to huggingface and make a request access for the models that we use in our work. Then run this command:
```bash 
pip install -r requirements.txt
```

### Create representation 
You can now extract the representations of your given model by running the following command: 

```bash 
python create_representation_dataset.py --model_name [model_name]
```
This creates a dataset/representations folder, which contains the representations under the given model path. 

### Run Self-Organizing Map
To train a simple 4x4 SOM and save the directions, you can run this command: 

```bash 
python som_generate_directions.py --som_x 4 --som_y 4 --layer 13 --sigma 0.33 --lr 0.01 --layer [l^*] --model_name [model_name]
```
Checkout the allowed args to see how to get the best out of your SOM! 

### BO search 
After the generation of the directions, you can run a BO search to find the best set of k direction to be ableated:

```bash 
python optuna_search.py --directions_path [path/to/directions] --model_name [model_name] --trials 512 --search_space 7 --search_bound 16
```

### Evaluate on test set
Finally, to evaluate the best set of directions on the test set, you can run the evaluation on the test set specifing the ids of the best directions:

```bash 
python orthogonalize.py --directions_path [path/to/directions] --model_name [model_name] --dir_ids [0, 1, 2]
```
and then to evaluate with the HarmBench Judge:
```bash 
python eval_jailbreaks.py --completions_path [path/to/completions]
```