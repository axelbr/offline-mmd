import itertools
from typing import Callable

import jax.random
from gymnax import EnvParams
from gymnax.environments.environment import Environment
import jax
import jax.numpy as jnp

import utils
from utils import Timestep



def get_dataset_mf(dataset: Timestep, env_params):
    dataset = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), dataset)
    mf = jnp.zeros_like(env_params.mean_field)
    mf = mf.at[dataset.state.time, dataset.state.x, dataset.state.y].add(1.0)
    mf = mf / mf.sum((1,2), keepdims=True)
    mf = jnp.where(jnp.isnan(mf), 0, mf)
    return mf


def make_mis_estimator(
        env: Environment, 
        env_params: EnvParams,
        dataset: Timestep, 
        policy_fn,
    ) -> Callable:

    def four_rooms_marginalized_importance_sampling(key, train_state) -> jnp.ndarray:
        policy = policy_fn(train_state)
        states = jnp.stack([dataset.state.time, dataset.state.x, dataset.state.y], axis=-1)
        dummy_key = jax.random.PRNGKey(0)
        batch_policy = lambda obs: jax.vmap(policy, in_axes=(None, 0))(dummy_key, obs)[1]

        def get_log_probs(_, data):
            obs, action = data
            pi = batch_policy(obs)
            return _, pi.log_prob(action)
        
        if dataset.obs.shape[0] > 10000:
            batch_obs = dataset.obs.reshape(dataset.obs.shape[0] // 1000, 1000, *dataset.obs.shape[1:])
            batch_action = dataset.action.reshape(dataset.action.shape[0] // 1000, 1000, *dataset.action.shape[1:])
            _, action_log_prob_pi = jax.lax.scan(get_log_probs, None, (batch_obs, batch_action))
            action_log_prob_pi = action_log_prob_pi.reshape(dataset.action.shape[0], dataset.action.shape[1])
        else:
            action_log_prob_pi = batch_policy(dataset.obs).log_prob(dataset.action)
        unique, indices, counts = jnp.unique(states.reshape(-1, 3), axis=0, return_counts=True, return_index=True)

        emp_state_dist = jnp.zeros(env_params.mean_field.shape, dtype=jnp.float32)
        emp_state_dist = emp_state_dist.at[unique[:, 0], unique[:, 1], unique[:, 2]].set(counts)
        emp_state_dist = emp_state_dist / emp_state_dist.sum((1, 2), keepdims=True)
        emp_state_dist = jnp.where(jnp.isnan(emp_state_dist), 0, emp_state_dist)

        est_mf = jnp.zeros_like(env_params.mean_field, dtype=jnp.float32)
        est_mf = est_mf.at[0].set(emp_state_dist[0])

        all_states = jnp.array(list(itertools.product(jnp.arange(est_mf.shape[1]), jnp.arange(est_mf.shape[2]))))

        def compute_est(t, x, y, mf_t):
            mask = (dataset.state.x[:, t + 1] == x) & (dataset.state.y[:, t + 1] == y)
            batch_t = jax.tree.map(lambda x: x[:, t], dataset)
            states = jnp.stack([batch_t.state.time, batch_t.state.x, batch_t.state.y], axis=-1)
            est_state_dist_t = mf_t[states[:, 1], states[:, 2]]
            emp_state_dist_t = emp_state_dist[t, states[:, 1], states[:, 2]]
            state_dist_ratio = est_state_dist_t / emp_state_dist_t
            state_dist_ratio = jnp.where(jnp.isnan(state_dist_ratio), 0, state_dist_ratio)
            action_log_prob_pi_t = action_log_prob_pi[:, t]
            action_ratio = jnp.exp(action_log_prob_pi_t - batch_t.action_log_prob)
            ratio = action_ratio * state_dist_ratio * mask
            est = jnp.sum(ratio * emp_state_dist[t + 1, x, y]) / mask.sum()
            est = jnp.where(jnp.isnan(est), 0, est)
            return est

        def compute_mf_t(mf_t, t):
            ts = jnp.full((all_states.shape[0],), t)
            xs = all_states[:, 0]
            ys = all_states[:, 1]
            mf_tp1 = jax.vmap(compute_est, in_axes=(0,0,0, None))(ts, xs, ys, mf_t)
            mf_tp1 = mf_tp1.reshape(est_mf.shape[1], est_mf.shape[2])
            return mf_tp1, mf_tp1

        _, computed_mf = jax.lax.scan(compute_mf_t, est_mf[0], jnp.arange(env_params.time_horizon-1))
        est_mf = jnp.concatenate([est_mf[:1], computed_mf], axis=0)
        est_mf = est_mf / est_mf.sum((1, 2), keepdims=True)
        est_mf = jnp.where(jnp.isnan(est_mf), 0, est_mf)

        return est_mf
    
    return four_rooms_marginalized_importance_sampling


def make_mean_field_generator(env: Environment, env_params: EnvParams, policy_fn):
    def compute_mean_field(key, train_state) -> jnp.ndarray:
        policy = policy_fn(train_state)
        num_states = env.state_space(env_params).n
        num_actions = env.action_space(env_params).n
        key = jax.random.PRNGKey(0)

        sas_tuple = jnp.stack(jnp.meshgrid(jnp.arange(num_states), jnp.arange(num_actions), jnp.arange(num_states)), axis=-1).reshape(-1, 3)
        states, actions, next_states = sas_tuple[:, 0], sas_tuple[:, 1], sas_tuple[:, 2]

        def compute_mf(mf, t, state, action, next_state):
            env_state = env.get_env_state(t-1, state)
            next_env_state = env.get_env_state(t, next_state)
            obs = env.get_obs(env_state, env_params, key)
            _, pi = policy(key, obs)
            mu_t = mf[state]
            next_state_density = jnp.zeros(num_states)
            mu_tp1 = mu_t * pi.prob(action) * env.dynamics(env_state, action, next_env_state)
            next_state_density = next_state_density.at[next_state].set(mu_tp1)
            return next_state_density

        def loop(carry, _):
            t, mf = carry
            ts = jnp.full(len(states), t)
            densities = jax.vmap(compute_mf, in_axes=(None, 0, 0, 0, 0))(mf, ts, states, actions, next_states)
            mf_t = densities.sum(0)
            return (t+1, mf_t), mf_t

        mf_0 = env_params.initial_state_distribution.reshape(-1, )
        initial = (1, mf_0)
        _, mf = jax.lax.scan(loop, initial, length=env.time_horizon-1)
        mf = jnp.concatenate([mf_0.reshape(1, -1), mf], axis=0)
        mf = mf.reshape(env_params.mean_field.shape)
        return mf

    return compute_mean_field