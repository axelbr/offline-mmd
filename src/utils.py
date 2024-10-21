import pickle
import chex
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import distrax
from typing import Callable
import flax.linen as nn
from gymnax import EnvState, EnvParams
from gymnax.environments.environment import Environment
import optax

Policy = Callable[[jax.random.PRNGKey, jnp.ndarray], jnp.ndarray | tuple[jnp.ndarray, distrax.Distribution]]


@chex.dataclass
class Transition:
    obs: jnp.ndarray
    state: EnvState
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    next_state: EnvState
    done: jnp.ndarray
    action_log_prob: jnp.ndarray


@chex.dataclass
class Timestep:
    obs: chex.Array
    state: EnvState
    action: chex.Array
    done: chex.Array
    reward: chex.Array
    action_log_prob: chex.Array


def get_softmax_policy(train_state: TrainState, temperature=1.0, return_dist=False) -> Policy:
    def policy_fn(key, obs):
        q_values = train_state.apply_fn(train_state.params, obs)
        dist = distrax.Categorical(logits=q_values / temperature)
        action = dist.sample(seed=key)
        return (action, dist) if return_dist else action

    return policy_fn


def get_epsilon_greedy_policy(train_state: TrainState, epsilon=0.0, return_dist=False) -> Policy:
    def policy_fn(key, obs):
        q_values = train_state.apply_fn(train_state.params, obs)
        dist = distrax.EpsilonGreedy(q_values, epsilon=epsilon)
        action = dist.sample(seed=key)
        return (action, dist) if return_dist else action

    return policy_fn

def get_random_policy(env: Environment, env_params: EnvParams = None, return_dist=False) -> Policy:
    env_params = env_params or env.default_env_params
    def policy_fn(key, obs):
        key, action_key = jax.random.split(key)
        dist = distrax.Categorical(probs=jnp.ones(env.action_space(env_params).n) / env.action_space(env_params).n)
        action = dist.sample(seed=action_key)
        return (action, dist) if return_dist else action
    return policy_fn


def load_checkpoint(path: str, q_network: nn.Module = None) -> TrainState:
    with open(path, "rb") as f:
        params = pickle.load(f)
    return TrainState.create(params=params, apply_fn=q_network.apply if q_network else None, tx=optax.adam(1e-3))

def evaluate(key: jax.random.PRNGKey, config, policy: Policy, env: Environment, env_params: EnvParams, num_episodes: int) -> tuple[Timestep, dict]:
    def step(carry, _):
        key, obs, env_state = carry
        key, action_key = jax.random.split(key)
        action, pi = policy(action_key, obs)
        key, step_key = jax.random.split(key)
        next_obs, next_env_state, reward, done, _ = jax.vmap(env.step, in_axes=(0,0,0,None))(
            jax.random.split(step_key, config["NUM_ENVS"]), env_state, action, env_params
        )
        timestep = Timestep(
            obs=obs,
            state=env_state,
            action=action,
            reward=reward,
            done=done,
            action_log_prob=pi.log_prob(action)
        )
        return (key, next_obs, next_env_state), jax.tree.map(jnp.squeeze, timestep)

    key, reset_key = jax.random.split(key)
    reset_key = jax.random.split(reset_key, config["NUM_ENVS"])
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
    num_steps = num_episodes * env_params.time_horizon // config["NUM_ENVS"]
    carry, trajectory_batch = jax.lax.scan(step, (key, obs, env_state), None, length=num_steps)
    trajectory_batch = jax.tree.map(lambda x: x.reshape((num_episodes, env_params.time_horizon, -1)).squeeze(), trajectory_batch)
    returns = trajectory_batch.reward.sum(-1)

    metrics = {
        "return": returns
    }
    return trajectory_batch, metrics
