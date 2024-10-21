from typing import Callable, NamedTuple, Tuple

import chex
import distrax
import jax.random
import optax
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax.training.train_state import TrainState
from gymnax import EnvParams, EnvState
from gymnax.environments.environment import Environment
import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
import flashbax as fbx
from gymnax.wrappers import LogWrapper

from utils import get_epsilon_greedy_policy


class OnlineMirrorDescentTrainState(TrainState):
    timesteps: int
    iterations: int
    n_updates: int
    prev_q_network_params: optax.Params
    target_q_network_params: optax.Params
    buffer_state: TrajectoryBufferState
    env_state: EnvState

@chex.dataclass
class Transition:
    obs: jnp.ndarray
    state: EnvState
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray

def make_learner(
        q_network: nn.Module,
        env: Environment, 
        env_params: EnvParams = None,
        buffer_size: int = 100000,
        buffer_batch_size: int = 32,
        num_updates: int = 1000,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_finish: float = 0.1,
        epsilon_anneal_time: int = 100000,
        temperature: float = 1.0,
        alpha: float = 1.0,
        discount_factor: float = 0.99,
        use_mirror_update: bool = True,
        use_dqn: bool = False,
        target_update_interval: int = 1000,
        training_starts: int = 1000,
        training_interval: int = 4,
        num_envs: int = 1,

):

    env_params = env_params or env.default_params
    act_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    env = LogWrapper(env)

    batch_reset = jax.vmap(env.reset, in_axes=(0, None))
    batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    buffer = fbx.make_flat_buffer(
        max_length=buffer_size,
        min_length=buffer_batch_size,
        sample_batch_size=buffer_batch_size,
        add_sequences=False,
        add_batch_size=num_envs,
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )
    epsilon = optax.schedules.linear_schedule(
        init_value=epsilon_start,
        end_value=epsilon_finish,
        transition_steps=epsilon_anneal_time
    )

    def init(key: jax.random.PRNGKey) -> OnlineMirrorDescentTrainState:
      
        key, init_key = jax.random.split(key)
        dummy_obs = obs_space.sample(jax.random.PRNGKey(0))
        q_network_params = q_network.init(init_key, dummy_obs)
        opt = optax.adam(lr)
        key, reset_key = jax.random.split(key, 2)

        obs, state = batch_reset(
            jax.random.split(reset_key, num_envs), env_params
        )
        buffer_state = buffer.init(Transition(
            obs=obs_space.sample(jax.random.PRNGKey(0)),
            state=jax.tree_map(lambda x: x[0], state.env_state),
            action=jnp.zeros((), dtype=jnp.int32),
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=jnp.bool_)
        ))
        return OnlineMirrorDescentTrainState.create(
            timesteps=0,
            iterations=0,
            n_updates=0,
            apply_fn=q_network.apply,
            params=q_network_params,
            target_q_network_params=jax.tree.map(jnp.copy, q_network_params),
            prev_q_network_params=jax.tree.map(jnp.copy, q_network_params),
            tx=opt,
            env_state=state,
            buffer_state=buffer_state
        )

    def _update_params(train_state: OnlineMirrorDescentTrainState, batch) -> tuple[OnlineMirrorDescentTrainState, float]:
        r_t = batch.first.reward
        s_t = batch.first.obs
        a_t = batch.first.action
        s_tp1 = batch.second.obs
        d_t = batch.first.done
        tau = temperature
        gamma = discount_factor
        q_network = train_state.apply_fn

        prev_q_t = q_network(train_state.prev_q_network_params, s_t)
        prev_q_tp1 = q_network(train_state.prev_q_network_params, s_tp1)
        target_q_tp1 = q_network(train_state.target_q_network_params, s_tp1)
        pi_t = distrax.Categorical(logits=prev_q_t / tau)
        pi_tp1 = distrax.Categorical(logits=prev_q_tp1 / tau)
        if use_mirror_update:
            soft_q_values = jnp.sum(pi_tp1.probs * (target_q_tp1 - tau * pi_tp1.logits), axis=-1)
            target = r_t + alpha * tau * pi_t.log_prob(a_t) + gamma * (1.0 - d_t) * soft_q_values
        elif use_dqn:
            target = r_t + gamma * (1.0 - d_t) * jnp.max(target_q_tp1, axis=-1)
        else:
            pi_target = distrax.Categorical(logits=target_q_tp1 / tau)
            target = r_t + gamma * (1.0 - d_t) * jnp.sum(pi_target.probs * target_q_tp1, axis=-1)

        def loss_fn(params):
            q_values = q_network(params, s_t)
            q_values = jnp.take_along_axis(q_values, a_t[:, None], axis=-1).squeeze(1)
            return jnp.mean(optax.l2_loss(q_values, target))

        loss, grad = jax.value_and_grad(loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grad)
        train_state = train_state.replace(
            n_updates=train_state.n_updates + 1
        )
        return train_state, loss

    def update(key: jax.random.PRNGKey, train_state: OnlineMirrorDescentTrainState, env_params: EnvParams = None) -> tuple[OnlineMirrorDescentTrainState, dict]:
        env_params = env_params or env.default_params
        key, action_key, step_key = jax.random.split(key, 3)
        policy = get_epsilon_greedy_policy(train_state, epsilon=epsilon(train_state.timesteps))
        obs = jax.vmap(env.get_obs, in_axes=(0, None))(train_state.env_state.env_state, env_params)
        action = jax.vmap(policy)(jax.random.split(action_key, num_envs), obs)
        _, next_env_state, reward, done, info = batch_step(
            jax.random.split(step_key, num_envs), train_state.env_state, action, env_params
        )
        timestep = Transition(
            obs=obs,
            state=train_state.env_state.env_state,
            action=action,
            reward=reward,
            done=done
        )
        buffer_state = buffer.add(train_state.buffer_state, timestep)


        do_update = (
                buffer.can_sample(buffer_state)
                & (train_state.timesteps >= training_starts)
                & (train_state.timesteps % training_interval == 0)
        )
        key, sample_key = jax.random.split(key)
        batch = buffer.sample(buffer_state, sample_key).experience
        train_state, loss = jax.lax.cond(
            do_update,
            _update_params,
            lambda ts, _: (train_state, 0.0),
            train_state, batch
        )
        new_target_network_params = optax.periodic_update(
            train_state.params,
            train_state.target_q_network_params,
            steps=train_state.timesteps,
            update_period=target_update_interval
        )

        train_state = train_state.replace(
            target_q_network_params=new_target_network_params,
            timesteps=train_state.timesteps + num_envs,
            env_state=next_env_state,
            buffer_state=buffer_state
        )
        metrics = {
            "loss": loss,
            "epsilon": epsilon(train_state.timesteps),
            "timesteps": train_state.timesteps,
            "returns": info["returned_episode_returns"].mean()
        }
        return train_state, metrics


    def train(key: jax.random.PRNGKey, train_state: OnlineMirrorDescentTrainState, env_params: EnvParams = None) -> tuple[OnlineMirrorDescentTrainState, dict]:
        def _loop_fn(carry, _):
            key, train_state = carry
            key, train_key = jax.random.split(key)
            train_state, metrics = update(train_key, train_state, env_params)
            return (key, train_state), metrics

        key, reset_key = jax.random.split(key, 2)
        _, state = batch_reset(
            jax.random.split(reset_key, num_envs), env_params
        )
        buffer_state = buffer.init(Transition(
            obs=obs_space.sample(jax.random.PRNGKey(0)),
            state=jax.tree_map(lambda x: x[0], state.env_state),
            action=jnp.zeros((), dtype=jnp.int32),
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=jnp.bool_)
        ))

        train_state = train_state.replace(
            prev_q_network_params=jax.tree.map(jnp.copy, train_state.params),
            target_q_network_params=jax.tree.map(jnp.copy, train_state.params),
            env_state=state,
            buffer_state=buffer_state
        )
        key, train_key = jax.random.split(key)
        num_steps = num_updates * training_interval // num_envs
        (_, train_state), metrics = jax.lax.scan(_loop_fn, (train_key, train_state), None, length=num_steps)
        train_state = train_state.replace(
            iterations=train_state.iterations + 1
        )
        return train_state, metrics

    return NamedTuple("Learner", [
        ("init", Callable[[jax.random.PRNGKey], OnlineMirrorDescentTrainState]),
        ("train", Callable[[jax.random.PRNGKey, OnlineMirrorDescentTrainState, EnvParams], Tuple[OnlineMirrorDescentTrainState, dict]]),
    ])(init, train)
