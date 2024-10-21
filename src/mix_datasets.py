import functools
import json
import json
import pickle
import flax.linen as nn
import warnings
import chex
import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

import distribution_estimation
from train import value_iteration
import utils
from envs import get_map, FourRoomEnv, FourRoomEnvParams

from utils import Timestep

warnings.filterwarnings("ignore")

"""
Script to generate datasets for the offline RL experiments. The script samples datasets from the base datasets and computes the quality metrics for each dataset.
"""

def sample_datasets(key: jax.random.PRNGKey, datasets: list[Timestep], num_datasets: int, sizes: list[int]) -> list[Timestep]:
    sampled_datasets = []
    for _ in range(num_datasets):
        samples = []
        for dataset in datasets:
            key, fraction_key = jax.random.split(key)
            fraction = jax.random.uniform(fraction_key, (1,)).item()
            num_samples = int(fraction * dataset.reward.shape[0])
            key, idx_key = jax.random.split(key)
            idx = jax.random.choice(idx_key, jnp.arange(dataset.reward.shape[0]), (num_samples,), replace=False)
            sample = jax.tree_map(lambda x: x[idx], dataset)
            samples.append(sample)
        key, size_key = jax.random.split(key)
        sample = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *samples)
        dataset_size = jax.random.choice(size_key, sizes, (1,)).item()
        dataset_size = min(dataset_size, sample.reward.shape[0])
        
        key, idx_key = jax.random.split(key)
        idx = jax.random.choice(idx_key, jnp.arange(sample.reward.shape[0]), (dataset_size,), replace=False)
        sample = jax.tree_map(lambda x: x[idx], sample)
        sampled_datasets.append(sample)
    return sampled_datasets


def compute_dataset_quality(cfg: DictConfig, dataset: Timestep, env: FourRoomEnv, env_params: FourRoomEnvParams, lower_bound: chex.Array, upper_bound: chex.Array) -> dict:
    timesteps = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), dataset)
    states = jnp.stack([timesteps.state.time, timesteps.state.x, timesteps.state.y], axis=-1)
    actions = dataset.action.reshape(-1,1)
    # state action coverage
    num_states = env.state_space(env_params).n
    num_actions = env.action_space(env_params).n
    unique_state_actions = jnp.unique(jnp.concatenate([states, actions], axis=-1), axis=0)
    unique_states = jnp.unique(states, axis=0)
    state_action_coverage = unique_state_actions.shape[0] / (num_states * num_actions * env_params.time_horizon)
    state_coverage = unique_states.shape[0] / (num_states * env_params.time_horizon)

    # trajectory quality
    returns = jnp.sum(dataset.reward, axis=-1).mean()
    trajectory_quality = (returns - lower_bound) / (upper_bound - lower_bound)
    trajectory_quality = jnp.clip(trajectory_quality, 0, 1)

    return {
        "state_action_coverage": state_action_coverage,
        "state_coverage": state_coverage,
        "trajectory_quality": trajectory_quality
    }

def generate_datasets(key, cfg: DictConfig, env: FourRoomEnv) -> list[Timestep]:
    datasets = []
    for path in cfg.base_datasets:
        with open(path, "rb") as f:
            dataset = pickle.load(f)
            datasets.append(dataset)
    

    key, sample_key = jax.random.split(key)
    q_network = nn.Sequential([
        nn.Dense(128),
        nn.relu,
        nn.Dense(128),
        nn.relu,
        nn.Dense(128),
        nn.relu,
        nn.Dense(5)
    ])
    expert_checkpoint = utils.load_checkpoint(cfg.expert_checkpoint, q_network=q_network)
    env_params = env.default_params

    random_mf_gen = distribution_estimation.make_mean_field_generator(
        env=env,
        env_params=env_params,
        policy_fn=lambda _: utils.get_random_policy(env, env_params, return_dist=True),
    )
    expert_mf_gen = distribution_estimation.make_mean_field_generator(
        env=env,
        env_params=env_params,
        policy_fn=functools.partial(utils.get_softmax_policy, temperature=cfg.algorithm.temperature, return_dist=True)
    )
    random_mf = random_mf_gen(jax.random.PRNGKey(0), None)
    expert_mf = expert_mf_gen(jax.random.PRNGKey(0), expert_checkpoint)
    random_policy = utils.get_random_policy(env, env_params, return_dist=True)
    expert_policy = utils.get_softmax_policy(expert_checkpoint, cfg.algorithm.temperature, return_dist=True)
    random_policy_value, _ = value_iteration(env, env_params.replace(mean_field=random_mf), policy=random_policy)
    expert_policy_value, _ = value_iteration(env, env_params.replace(mean_field=expert_mf), policy=expert_policy)

    sampled_datasets = sample_datasets(
        key=sample_key,
        datasets=datasets,
        num_datasets=cfg.num_datasets,
        sizes=jnp.array([200, 400, 500, 750, 1000, 2000, 3000, 5000, 10000, 20000, 50000])
    )
    datasets.extend(sampled_datasets)

    dataset_qualities = []
    for i, dataset in enumerate(datasets):
        quality_metrics = compute_dataset_quality(
            cfg=cfg,
            dataset=dataset,
            env=env,
            env_params=env_params,
            lower_bound=random_policy_value,
            upper_bound=expert_policy_value
        )
        dataset_qualities.append(quality_metrics)
        with open(f"{cfg.dataset_dir}/{i:03d}.pkl", "wb") as f:
            pickle.dump(dataset, f)

    dataset_qualities = jax.tree.map(lambda *x: jnp.stack(x).tolist(), *dataset_qualities)
    with open(f"{cfg.dataset_dir}/qualities.json", "w") as f:
        json.dump(dataset_qualities, f)

@hydra.main(version_base=None, config_path="config", config_name="mix_datasets")
def main(cfg: OmegaConf) -> None:
    key = jax.random.PRNGKey(cfg.seed)
    env = FourRoomEnv(task=cfg.task)
    generate_datasets(key, cfg, env)
    

if __name__ == '__main__':
    main()