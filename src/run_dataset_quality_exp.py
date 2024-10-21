import copy
import glob
import json
import os
import pickle
import hydra

from joblib import Parallel, delayed
import numpy as np
from omegaconf import OmegaConf

def train_on_dataset(cfg, logdir, dataset_path, index):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index % 4)
    import jax
    from compute_stats import compute_metric_stats
    from envs import FourRoomEnv
    from train import make_train

    print(f"Training on {dataset_path}, using GPU {jax.devices()}")

    key = jax.random.PRNGKey(cfg.seed)
    env = FourRoomEnv(task=cfg.task)
    
    os.makedirs(logdir, exist_ok=True)
    with open(f"{logdir}/config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f)

    cfg.logdir = logdir
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    train_fn = make_train(cfg, env=env, dataset=dataset)
    
    keys = jax.random.split(key, cfg.num_seeds)

    with jax.disable_jit(not cfg.use_jit):
        #train_fn = jax.jit(train_fn)
        metrics = jax.vmap(train_fn)(keys)

    scalar_metrics = {k: v for k, v in metrics.items() if k.startswith("metric")}

    stats = compute_metric_stats(scalar_metrics)
    with open(f"{logdir}/results.json", "w") as f:
        json.dump({k: v.tolist() for k, v in metrics.items()}, f)
    with open(f"{logdir}/stats.json", "w") as f:
        json.dump(stats, f)

def cql_dataset_sensitiviy_experiment(cfg):
    with open(cfg.dataset_dir + "/qualities.json", "r") as f:
        quality_metrics = json.load(f)
    coverages = np.array(quality_metrics["state_action_coverage"])
    dataset_paths = sorted(glob.glob(cfg.dataset_dir + "/*.pkl"))
    datasets_per_coverage = {}
    all_datasets = []
    for cov in cfg.coverage_levels:
        indices = np.argsort(np.abs(coverages - cov))[:cfg.num_datasets_per_coverage]
        datasets_per_coverage[cov] = [dataset_paths[i] for i in indices]
        all_datasets.extend(datasets_per_coverage[cov])
        
    runner = Parallel(n_jobs=20)
    train_fn = delayed(train_on_dataset)
    jobs = []

    i = 0
    for cql_weight in cfg.cql_weights:
        for cov in cfg.coverage_levels:
            for dataset in datasets_per_coverage[cov]:
                logdir = f"logs/{cfg.task}_{cfg.experiment_name}/cql_weight_{cql_weight}_coverage_{cov}/{os.path.basename(dataset).split('.')[0]}"
                job_cfg = copy.deepcopy(cfg)
                job_cfg.algorithm.cql_weight = cql_weight
                job_cfg.gpu_id = i % 4
                jobs.append(train_fn(job_cfg, logdir, dataset, i))
                print(logdir)
                print(OmegaConf.to_yaml(job_cfg))
                i += 1
    print(f"Running {len(jobs)} jobs")
    print(f"Num duplicates: {len(all_datasets) - len(set(all_datasets))}")
    runner(jobs)


def dataset_quality_ablation(cfg):
    datasets = sorted(glob.glob(cfg.dataset_dir + "/*.pkl"))
    runner = Parallel(n_jobs=20)
    train_fn = delayed(train_on_dataset)
    jobs = []
    for i, dataset in enumerate(datasets):
        logdir = f"logs/{cfg.task}_{cfg.experiment_name}/{os.path.basename(dataset).split('.')[0]}"
        jobs.append(train_fn(cfg, logdir, dataset, i))
    runner(jobs)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg):
    if cfg.experiment_name.startswith("cql_dataset_sensitivity"):
        cql_dataset_sensitiviy_experiment(cfg)
    elif cfg.experiment_name == "dataset_quality_ablation":
        dataset_quality_ablation(cfg)
    

if __name__ == "__main__":
    main()