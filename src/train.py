import functools
import json
import json
import os
import pickle
import time
import flax.linen as nn
import warnings
from typing import Tuple, Dict, Callable

import chex
import distrax
import hydra

import numpy as np
from omegaconf import OmegaConf
import scipy.stats as stats
from flax.training.train_state import TrainState
from gymnax.environments.environment import Environment
from matplotlib import pyplot as plt

import distribution_estimation
import utils
from envs import FourRoomEnv, FourRoomEnvParams
from utils import Timestep
from collections import defaultdict
import jax
import jax.numpy as jnp
import offline_mirror_descent
import online_mirror_descent

warnings.filterwarnings("ignore")

Policy = Callable[[jax.random.PRNGKey, chex.Array], chex.Array | Tuple[chex.Array, distrax.Distribution]]

class LoggingCallback:

    def __init__(self, env, logdir: str, log_to_console: bool = True, show_plots: bool = True):
        if hasattr(env, "map"):
            self.map = env.map
        else:
            self.map = None
        self.logdir = logdir
        self.iteration_logs = defaultdict(list)
        self.mean_fields = defaultdict(list)
        self.log_to_console = log_to_console
        self.show_plots = show_plots
        self.mean_field_plots = defaultdict(list)

    def compute_stats(self, values: jnp.ndarray) -> Dict[str, float]:
        """
        Compute statistics of the values.
        :param values: Array of values. Shape (num_episodes,)
        """
        if values.shape == () or len(values) == 1:
            return {
                "mean": values.item(),
                "std": 0,
                "min": values.item(),
                "max": values.item()
            }
        else:
            return {
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
                "ci95": stats.t.interval(0.95, len(values) - 1, loc=jnp.mean(values), scale=stats.sem(values))[1] - jnp.mean(values)
            }

    def draw_mean_field(self, mf):
        colormap = plt.colormaps["YlGnBu"]
        T = mf.shape[0]
        t0, t1,  t2, t4 = 0, 5, int(T*0.5), T-1
        imgs = []
        for t in [t0, t1, t2, t4]:
            vals = mf[t]  # .clip(0, 0.02)
            vals = vals / 0.02
            img = colormap(vals)
            # img[state.x, state.y] = [0, 1, 0, 1]
            if self.map is not None:
                img[self.map == 0] = [0, 0, 0, 1]
            imgs.append(img)

        img = np.concatenate(imgs, axis=1)
        return img

    def on_iteration_end(self, train_state: TrainState, metrics: dict, seed: int = None) -> None:
        if seed is not None:
            seed = seed.item()
            logdir = f"{self.logdir}/{seed:02d}"
        else:
            logdir = self.logdir
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(f"{logdir}/mean_fields", exist_ok=True)
        os.makedirs(f"{logdir}/checkpoints", exist_ok=True)


        stats = {k: self.compute_stats(v) for k, v in metrics.items() if k.startswith("metric")}
        self.iteration_logs[seed].append(stats)
        mean_fields = {
            "gt_mf": metrics["mean_field/ground_truth"],
        }
        if "est_mf" in metrics:
            mean_fields["est_mf"] = metrics["estimated"]
        self.mean_fields[seed].append(mean_fields)

        with open(f"{logdir}/checkpoints/{train_state.iterations:03d}.pkl", "wb") as f:
            pickle.dump(train_state.params, f)

        if self.log_to_console:
            values = ", ".join([f"{k}={v['mean']:.2f}" for k, v in stats.items()])
            string = f"Iteration {train_state.iterations:03d}: {values}"
            print(string)

        if self.show_plots:
            imgs = [
                self.draw_mean_field(metrics["mean_field/ground_truth"]),
            ]
            if "mean_field/estimated" in metrics:
                imgs.append(np.zeros((1, imgs[0].shape[1], 4)))
                imgs.append(self.draw_mean_field(metrics["mean_field/estimated"]))
            img = np.concatenate(imgs, axis=0)
            fig, ax = plt.subplots()
            ax.imshow(img)
            fig.savefig(f"{logdir}/mean_fields/{train_state.iterations:03d}.png")
            plt.show()



def update_dataset(key: jax.random.PRNGKey, dataset: Timestep, env: FourRoomEnv, env_params: FourRoomEnvParams) -> Timestep:
    states = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), dataset.state)
    actions = dataset.action.reshape(-1)
    dataset_size = actions.shape[0]
    _, rewards, _, _, _ = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))(
        jax.random.split(key, dataset_size), states, actions, env_params
    )
    rewards = rewards.reshape(dataset.reward.shape)
    return dataset.replace(reward=rewards)


def value_iteration(env: FourRoomEnv, env_params: FourRoomEnvParams, policy: Policy = None, discount: float = 1.0) -> jnp.ndarray:
    T, S, A = env_params.time_horizon, env.state_space(env_params).n, env.action_space(env_params).n
    q_table = jnp.zeros((T, S, A))

    def reward_fn(t, s, a, s_next):
        env_state = env.get_env_state(t, s)
        next_env_state = env.get_env_state(1+1, s_next)
        return env.get_reward(env_state, a, next_env_state, env_params)

    def dynamics(t, s, a, s_next):
        env_state = env.get_env_state(t, s)
        next_env_state = env.get_env_state(t+1, s_next)
        return env.dynamics(env_state, a, next_env_state)

    def get_value(Q, t, s):
        if policy is not None:
            key = jax.random.PRNGKey(0)
            env_state = env.get_env_state(t, s)
            obs = env.get_obs(env_state, env_params)
            _, dist = policy(key, obs)
            return jnp.sum(dist.probs * Q[s])
        else:
            return Q[s].max(-1)

    def update(t, s, a, Q_tp1):
        def get_q_values(s_next):
            p = dynamics(t, s, a, s_next)
            r = reward_fn(t, s, a, s_next)
            return p * (r + discount * get_value(Q_tp1, t+1, s_next))
        q_values = jax.vmap(get_q_values)(jnp.arange(S)).sum(0)
        return q_values

    def update_table(q_table, t):
        q_table = jax.vmap(lambda s: jax.vmap(lambda a: update(t, s, a, q_table))(jnp.arange(A)))(jnp.arange(S))
        return q_table, q_table

    _, q_table = jax.lax.scan(update_table, q_table[T-1], jnp.arange(T-1, -1, -1))
    q_table = q_table[::-1]
    env_states = jax.vmap(env.get_env_state, in_axes=(None, 0))(0, jnp.arange(S))
    d_0 = jax.vmap(lambda s: env_params.initial_state_distribution[s.x, s.y])(env_states)
    q_values = jnp.sum(d_0 * q_table[0].max(-1))
    return q_values, q_table


def make_train(cfg: OmegaConf, env: Environment, dataset: Timestep = None):

    callback = LoggingCallback(env, cfg.logdir, log_to_console=True, show_plots=True)

    q_network = nn.Sequential([
        nn.Dense(128),
        nn.relu,
        nn.Dense(128),
        nn.relu,
        nn.Dense(128),
        nn.relu,
        nn.Dense(env.action_space(env.default_params).n)
    ])

    def evaluate_policy(key: jax.random.PRNGKey, policy, env_params: FourRoomEnvParams) -> dict:
        # Compute exploitability
        key, eval_key, br_key, br_eval_key, = jax.random.split(key, 4)
        num_envs = cfg.training.num_envs

        def collect_rollout(key, policy):
            def step(carry, _):
                key, obs, env_state = carry
                key, action_key = jax.random.split(key)
                action, pi = policy(action_key, obs)
                key, step_key = jax.random.split(key)
                next_obs, next_env_state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    jax.random.split(step_key, num_envs), env_state, action, env_params
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
            reset_key = jax.random.split(reset_key, cfg.training.num_envs)
            obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
            num_steps = cfg.evaluation.num_episodes * env.time_horizon // cfg.training.num_envs
            carry, trajectory_batch = jax.lax.scan(step, (key, obs, env_state), None, length=num_steps)
            trajectory_batch = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), trajectory_batch)
            trajectory_batch = jax.tree.map(lambda x: x.reshape(cfg.evaluation.num_episodes, env.time_horizon, -1).squeeze(), trajectory_batch)
            returns = trajectory_batch.reward.sum(-1).mean()

            metrics = {
                "return": returns
            }
            return trajectory_batch, metrics

        eval_traj, eval_metrics = collect_rollout(eval_key, policy)
        metrics = {
            "metric/return": eval_metrics["return"],
        }

        if cfg.environment == "four_rooms":
            policy_value, _ = value_iteration(env, env_params, policy=policy, discount=cfg.algorithm.discount_factor)
            br_value, _ = value_iteration(env, env_params, discount=cfg.algorithm.discount_factor)
            exploitability = br_value - policy_value
            metrics["metric/policy_value"] = policy_value
            metrics["metric/br_value"] = br_value
            metrics["metric/exploitability"] = exploitability

        #_, br_metrics = collect_rollout(br_key, br_policy)
        return eval_traj, metrics

    def train(key: jax.random.PRNGKey):

        SEED = key[1]
        env_params = env.default_params
        learner_fn = hydra.utils.call(cfg.algorithm)
        if cfg.offline:
            learner = learner_fn(
                q_network=q_network,
                env=env,
                env_params=env_params,
                dataset=dataset
            )
        else:
            learner = learner_fn(
                q_network=q_network,
                env=env,
                env_params=env_params
            )

        # Make best response learner
        gt_mean_field_estimator = distribution_estimation.make_mean_field_generator(
            env,
            env_params,
            policy_fn=functools.partial(utils.get_softmax_policy, temperature=cfg.algorithm.temperature, return_dist=True)
        )

        offline_estimator = distribution_estimation.make_mis_estimator(
            env,
            env_params,
            dataset,
            policy_fn=functools.partial(utils.get_softmax_policy, temperature=cfg.algorithm.temperature, return_dist=True)
        )

        def run_iteration(carry, _):
            key, ts, env_params = carry

            # Update policy
            policy = utils.get_softmax_policy(ts, temperature=cfg.algorithm.temperature, return_dist=True)

            # Compute new mean field
            key, mf_key, eval_key = jax.random.split(key, 3)
            mean_field = gt_mean_field_estimator(mf_key, ts)
            env_params = env_params.replace(mean_field=mean_field)

            # Train approximate best response against current mean field
            #key, init_key, train_key = jax.random.split(key, 3)
            #br_ts = br_learner.init(init_key).replace(params=ts.params)
            #br_ts, _ = br_learner.train(train_key, br_ts, env_params)

            # Evaluate current policy
            _, metrics = evaluate_policy(
                key=eval_key,
                policy=policy,
                env_params=env_params
            )

            if cfg.offline:
                # Estimate mean field for training
                key, mf_key = jax.random.split(key)
                estimated_mf = offline_estimator(mf_key, ts)
            else:
                # Use ground truth mean field for training
                estimated_mf = mean_field

            # Update current mean field
            env_params = env_params.replace(mean_field=estimated_mf)

            # Update Q network
            key, train_key = jax.random.split(key)
            ts, train_metrics = learner.train(train_key, ts, env_params)

            # Update logs
            logs = {
                **metrics,
                **{f"metric/{k}": jnp.mean(v) for k, v in train_metrics.items()},
                "mean_field/ground_truth": mean_field,
                "mean_field/estimated": estimated_mf
            }

            # Log iteration
            if cfg.num_seeds > 1:
                jax.debug.callback(callback.on_iteration_end, ts, logs, SEED)
            else:
                jax.debug.callback(callback.on_iteration_end, ts, logs)


            return (key, ts, env_params), metrics

        # Initialize train state
        key, init_key = jax.random.split(key)
        ts = learner.init(init_key)
        logs = []
        step = (key, ts, env_params)
        for _ in range(cfg.training.num_iterations):
            step, metrics = run_iteration(step, None)
            logs.append(metrics)

        logs = jax.tree.map(lambda *x: jnp.stack(x), *logs)

        return logs
    return train


def compute_metric_stats(metrics: dict) -> dict:

    def compute_seed_stats(x):
        mean = jnp.mean(x, 0)
        low, high = stats.t.interval(0.95, x.shape[0] - 1, loc=mean, scale=stats.sem(x, 0))
        return jax.tree.map(lambda x: np.array(x).tolist(), {
            "mean": mean,
            "std": jnp.std(x, 0),
            "ci_95_low": low,
            "ci_95_high": high
        })

    # Compute intervals over seeds
    data = jax.tree.map(compute_seed_stats, metrics)
    return data



@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: OmegaConf) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Trick to fix the gpu to train on
    gpu_id = cfg.gpu_id % cfg.num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import jax
    import jax.numpy as jnp
   
    key = jax.random.PRNGKey(cfg.seed)
    if cfg.environment == "four_rooms":
        env = FourRoomEnv(task=cfg.task)
    else:
        raise ValueError(f"Unknown environment: {cfg.environment}")

    if isinstance(cfg.dataset_path, str):
        dataset_path = [cfg.dataset_path]
    else:
        dataset_path = cfg.dataset_path
    datasets = []
    for path in dataset_path:
        with open(path, "rb") as f:
            dataset = pickle.load(f)
            datasets.append(dataset)
    
    dataset = jax.tree_map(lambda *x: jnp.concatenate(x, axis=0), *datasets)
    logdir = f"logs/{cfg.experiment_name}_{time.time()}"
    os.makedirs(logdir, exist_ok=True)
    with open(f"{logdir}/config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f)
   
    cfg.logdir = logdir
    train_fn = make_train(cfg, env=env, dataset=dataset)
    keys = jax.random.split(key, cfg.num_seeds)

    with jax.disable_jit(not cfg.use_jit):
        metrics = jax.vmap(train_fn)(keys)

    scalar_metrics = {k: v for k, v in metrics.items() if k.startswith("metric")}

    stats = compute_metric_stats(scalar_metrics)
    with open(f"{logdir}/results.json", "w") as f:
        json.dump({k: v.tolist() for k, v in metrics.items()}, f)
    with open(f"{logdir}/stats.json", "w") as f:
        json.dump(stats, f)
    




if __name__ == '__main__':
    main()
