from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from functools import partial
from typing import List
import random
import optuna
import torch
from optuna.distributions import IntDistribution

from models.language_models import Llama2_7b, Llama3_8b, Qwen_7b, Qwen_14b, Qwen2_3b, Qwen2_7b, Mistral7B_RR, Zephyr_R2D2, Llama2_13b, Gemma2_9b
from utils.ortho_utils import orthogonalize_weights
from directions_ablation import generate_and_save_hookfree_completions
from config import Config
from eval_jailbreaks import evaluate_jailbreak


# --------------------------------------------------------------------------- #
#  helper                                                                     #
# --------------------------------------------------------------------------- #
def extract_direction_prefix(path: str) -> str:
    m = re.search(r"((?:\w+_)?som[^\s/]+_layer\d{1,2})", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse prefix from {path}")
    return m.group(1)

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
    return models[model_name](device=device)


# --------------------------------------------------------------------------- #
#  Optuna sampler that guarantees unique dirs inside each trial               #
# --------------------------------------------------------------------------- #
class UniqueDirSampler(optuna.samplers.BaseSampler):
    """Resample dir_* finché non sono tutti diversi, poi delega al sampler base."""

    def __init__(self, base: optuna.samplers.BaseSampler, fixed_dirs: List[int]):
        self._base = base
        self._rand = random.Random()
        self._fixed_dirs = set(fixed_dirs) 


    def infer_relative_search_space(self, study, trial):
        return self._base.infer_relative_search_space(study, trial)

    def sample_relative(self, study, trial, search_space):
        params = self._base.sample_relative(study, trial, search_space)

        seen = set()
        for name, dist in search_space.items():
            if not name.startswith("dir_"):
                continue
            if name not in params:
                continue
            val = params[name]
            if val in seen:
                domain = range(dist.low, dist.high + 1)
                remaining = [v for v in domain if v not in seen]
                params[name] = self._rand.choice(remaining)
            seen.add(params[name])

        return params

    def reseed_rng(self):
        self._base.reseed_rng()
        self._rand.seed(self._base._rng.randint(0, 2**32 - 1))  # type: ignore[attr-defined]

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: IntDistribution,
    ):
    
        if not param_name.startswith("dir_"):
            return self._base.sample_independent(
                study, trial, param_name, param_distribution
            )

        tried = set(self._fixed_dirs)  # ← make a fresh copy
        tried.update(v for k, v in trial.params.items() if k.startswith("dir_"))
        domain = list(range(param_distribution.low, param_distribution.high + 1))

        val = self._base.sample_independent(study, trial, param_name, param_distribution)
        if val not in tried:
            return val

        remaining = [v for v in domain if v not in tried]
        return self._rand.choice(remaining)


# --------------------------------------------------------------------------- #
#  evaluation function                                                        #
# --------------------------------------------------------------------------- #
def evaluate_attack(
    dirs: List[int],
    model_name: str,
    directions_path: str,
    dataset_name: str,
    device: str,
    aux_name: str,
    eval_path: str,
) -> float:
    llm = get_model(model_name, device)
    cfg = Config(model_alias=model_name, model_path=model_name)

    local_aux = f"{aux_name}_{dataset_name}_{'_'.join(map(str, dirs))}"
    print(f">> evaluating dirs {dirs}")

    ortho_dirs = torch.load(directions_path)
    aux_name = f"weights_raw_dirs_{extract_direction_prefix(directions_path)}"
    print(">> ablating directions")
    for d in dirs:
        orthogonalize_weights(llm, ortho_dirs[d])


    completions = generate_and_save_hookfree_completions(
        cfg=cfg,
        folder="val_optim_completions",
        model_base=llm,
        dataset_name=dataset_name,
        aux_name=local_aux,
        return_completions=True,
    )
    del llm
    torch.cuda.empty_cache()

    os.makedirs(eval_path, exist_ok=True)
    result = evaluate_jailbreak(
        completions=completions,
        methodologies=["harmbench"],
        evaluation_path=f"{eval_path}/{local_aux}_results.json"
    )
    return float(result["harmbench_success_rate"])


# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #
def run_optimization(args):

    # set path 
    evaluation_path = f"./runs/{args.model_name}/val_optim_evaluations"

    aux_name = f"weights_raw_dirs_{extract_direction_prefix(args.directions_path)}"
    print(">> using RAW directions")


    bound, space = args.search_bound - 1, args.search_space
    assert bound + 1 >= space - 1, "Not enough distinct indices for the requested space"

    # ---- build the objective --------------------------------------------- #
    objective = partial(
        _objective,
        bound=bound,
        space=space,
        model_name=args.model_name,
        directions_path=args.directions_path,
        dataset_name=args.dataset_name,
        device='cuda:0',
        aux_name=aux_name,
        eval_path=evaluation_path,
        fixed_dirs=args.fixed_dirs
    )

    # ---- study with custom sampler (TPE under the hood) ------------------- #
    base_sampler = optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=args.trials // 4)
    sampler = UniqueDirSampler(base_sampler, fixed_dirs=args.fixed_dirs)
    study = optuna.create_study(study_name=aux_name, direction="maximize", sampler=sampler)
    # Check that we have enough directions to sample from
    total_needed = len(args.fixed_dirs) + (args.search_space - len(args.fixed_dirs))
    if total_needed > args.search_bound:
        raise ValueError(
            f"Cannot sample {total_needed} unique directions with only {args.search_bound} available. "
            f"Reduce search_space or fixed_dirs, or increase search_bound."
        )
    study.optimize(objective, n_trials=args.trials, n_jobs=1, show_progress_bar=True)

    # ---- save artefacts --------------------------------------------------- #
    base = os.path.splitext(os.path.basename(args.directions_path))[0]
    if not os.path.exists(f"./runs/{args.model_name}/optimizations"):
        os.makedirs(f"./runs/{args.model_name}/optimizations")
    with open(f"./runs/{args.model_name}/optimizations/{base}_optuna_study_{args.trials}.pkl", "wb") as f:
        pickle.dump(study, f)


    best_summary = {"best_params": study.best_params, "best_score": study.best_value}

    trial_details = []
    for t in study.trials:
        trial_details.append(
            {
                "number": t.number,
                "state": t.state.name,                # COMPLETE / FAIL / PRUNED
                "value": t.value,                     # None se FAIL/PRUNED
                "params": t.params,
                "start": t.datetime_start.isoformat() if t.datetime_start else None,
                "end":   t.datetime_complete.isoformat() if t.datetime_complete else None,
                "duration_sec": t.duration.total_seconds() if t.duration else None,
                "user_attrs": t.user_attrs,
                "system_attrs": t.system_attrs,
                "intermediate": dict(t.intermediate_values),  # step: value
            }
        )

    out_dir = f"./runs/{args.model_name}/optimizations"
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{out_dir}/{base}_optuna_results_t:{args.trials}_{args.dataset_name}.json" 
    with open(filename, "w") as fp:
        json.dump(
            {
                **best_summary,
                "n_trials": len(study.trials),
                "trials": trial_details,
            },
            fp,
            indent=2,
        )

    print("Best params :", study.best_params)
    print("Best score :", study.best_value)


# --------------------------------------------------------------------------- #
#  Optuna objective (define-by-run)                                           #
# --------------------------------------------------------------------------- #
def _objective(
    trial: optuna.Trial,
    *,
    bound: int,
    space: int,
    model_name: str,
    directions_path: str,
    dataset_name: str,
    device: str,
    aux_name: str,
    eval_path: str,
    fixed_dirs: List[int]
) -> float:
    dirs = fixed_dirs + [trial.suggest_int(f"dir_{i}", 0, bound) for i in range(space - len(fixed_dirs))]

    return evaluate_attack(
        dirs=dirs,
        model_name=model_name,
        directions_path=directions_path,
        dataset_name=dataset_name,
        device=device,
        aux_name=aux_name,
        eval_path=eval_path,
    )


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directions_path", required=True)
    p.add_argument("--dataset_name", default="harmbench_val")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model_name", required=True)
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--search_space", type=int, default=5)
    p.add_argument("--search_bound", type=int, default=3)
    p.add_argument("--fixed_dirs", nargs='+', type=int, default=[], help="List of fixed direction indices") 
    os.environ["CUDA_VISIBLE_DEVICES"] = p.parse_known_args()[0].device.split(":")[-1]
    run_optimization(p.parse_args())