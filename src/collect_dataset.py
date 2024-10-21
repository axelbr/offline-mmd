import functools
import os

import distrax
import jax
import jax.numpy as jnp
import pickle
import flax.linen as nn

from gymnax import EnvParams
from gymnax.environments.environment import Environment

import distribution_estimation
from envs import FourRoomEnv, FOUR_ROOMS
from utils import Timestep


def collect_episodes(key, config, env, env_params, policy, num_episodes: int):

    estimator = distribution_estimation.make_mean_field_generator(
        env=env, env_params=env_params, policy_fn=lambda _: policy
    )
    mf = estimator(jax.random.PRNGKey(0), None)
    env_params = env_params.replace(mean_field=mf)

    @jax.vmap
    def collect_timesteps(carry, _):
        key, state, obs = carry
        key, action_key = jax.random.split(key)
        action, dist = policy(action_key, obs)
        key, step_key = jax.random.split(key)
        next_obs, next_state, reward, done, _ = env.step(
            step_key, state, action, env_params
        )
        transition = Timestep(
            obs=obs,
            state=state,
            action=action,
            reward=reward,
            done=done,
            action_log_prob=dist.log_prob(action),
        )
        return (key, next_state, next_obs), jax.tree.map(jnp.squeeze, transition)
  
    key, reset_key = jax.random.split(key)
    reset_key = jax.random.split(reset_key, num_episodes)
    obs, state = jax.vmap(env.reset_env, in_axes=(0, None))(reset_key, env_params)
    step_keys = jax.random.split(key, num_episodes)
    num_time_steps = env_params.time_horizon
    _, timesteps = jax.lax.scan(
        collect_timesteps, (step_keys, state, obs), None, length=num_time_steps
    )
    timesteps = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), timesteps)
    
    return timesteps


def get_random_policy(env: Environment, env_params: EnvParams):
    def policy_fn(key, obs):
        logits = jnp.zeros((env.action_space(env_params).n,))
        dist = distrax.Categorical(logits=logits)
        act = dist.sample(seed=key)
        return act, dist

    return policy_fn


def get_checkpoint_policy(config: dict, env: Environment, env_params: EnvParams):

    qnet = nn.Sequential(
        [
            nn.Dense(128),
            nn.relu,
            nn.Dense(128),
            nn.relu,
            nn.Dense(128),
            nn.relu,
            nn.Dense(env.action_space(env_params).n),
        ]
    )

    def softmax_policy(key, obs, params, temperature=1.0):
        q_values = qnet.apply(params, obs)
        dist = distrax.Categorical(logits=q_values / temperature)
        action = dist.sample(seed=key)
        return (action, dist)

    with open(config["POLICY_CHECKPOINT"], "rb") as f:
        params = pickle.load(f)
    if config["POLICY"] == "checkpoint":
        temperature = config["POLICY_TEMPERATURE"]
        policy = functools.partial(
            softmax_policy, temperature=temperature, params=params
        )
    else:
        raise ValueError("Unknown policy")
    return policy


def collect_four_room_dataset(config):
    env = FourRoomEnv(task=config["TASK"])
    env_params = env.default_params
    if config["POLICY"] == "random":
        policy = get_random_policy(env, env_params)
    elif config["POLICY"] == "checkpoint":
        policy = get_checkpoint_policy(config, env, env_params)
    else:
        raise ValueError("Unknown policy")

    key = jax.random.PRNGKey(config["SEED"])
    key, run_key = jax.random.split(key)
    dataset = collect_episodes(
        config=config, env=env, env_params=env_params, key=run_key, policy=policy, num_episodes=config["NUM_EPISODES"]
    )
    return dataset


if __name__ == "__main__":
    config = {
        "SEED": 42,
        "ENV": "four_rooms",
        "TASK": "exploration", # exploration, navigation, labyrinth
        "POLICY": "checkpoint",
        "POLICY_CHECKPOINT": "datasets/policy_checkpoints/exploration_expert.pkl",        
        "POLICY_TEMPERATURE": 20.0,
        "NUM_EPISODES": 1,
        "DATASET_PATH": ".",
        "DATASET_NAME": "test",
    }

    os.makedirs(config["DATASET_PATH"], exist_ok=True)
    if config["ENV"] == "four_rooms":
        dataset = collect_four_room_dataset(config)
    else:
        raise ValueError("unknown env")

    policy_name = config["POLICY"]
    num_policies = 1
    if policy_name == "checkpoint":
        if isinstance(config["POLICY_CHECKPOINT"], list):
            policy_name = "_".join(
                [os.path.basename(p).split(".")[0] for p in config["POLICY_CHECKPOINT"]]
            )
            num_policies = len(config["POLICY_CHECKPOINT"])
        else:
            policy_name = f"{policy_name}_{os.path.basename(config['POLICY_CHECKPOINT']).split('.')[0]}"

    path = f"{config['DATASET_PATH']}/{config['DATASET_NAME']}.pkl"
    print(f"Saving dataset to {path}")

    with open(path, "wb") as f:
        pickle.dump(dataset, f)
