from typing import Callable, NamedTuple, Tuple

import distrax
import jax.random
import optax
from flax.training.train_state import TrainState
from gymnax import EnvParams
from gymnax.environments.environment import Environment
import jax
import jax.numpy as jnp
import flax.linen as nn

from utils import Timestep


class OfflineMirrorDescentTrainState(TrainState):
    timesteps: int
    iterations: int
    n_updates: int
    prev_q_network_params: optax.Params
    target_q_network_params: optax.Params


def make_learner(
        q_network: nn.Module,
        dataset: Timestep,
        env: Environment,
        env_params: EnvParams,
        lr: float,
        temperature: float,
        alpha: float,
        discount_factor: float,
        cql_weight: float,
        num_batches: int,
        batch_size: int,

):

    obs_shape = env.observation_space(env_params).shape
    
    def init(key: jax.random.PRNGKey) -> OfflineMirrorDescentTrainState:
        
        key, init_key = jax.random.split(key)
        dummy_obs = jnp.zeros(obs_shape, jnp.float32)
        q_network_params = q_network.init(init_key, dummy_obs)
        opt = optax.adam(lr)
        key, reset_key = jax.random.split(key, 2)


        return OfflineMirrorDescentTrainState.create(
            timesteps=0,
            iterations=0,
            n_updates=0,
            apply_fn=q_network.apply,
            params=q_network_params,
            target_q_network_params=jax.tree.map(jnp.copy, q_network_params),
            prev_q_network_params=jax.tree.map(jnp.copy, q_network_params),
            tx=opt
        )

    def update(key: jax.random.PRNGKey, train_state: OfflineMirrorDescentTrainState, batch: Timestep, env_params: EnvParams = None) -> tuple[OfflineMirrorDescentTrainState, dict]:
        r_t = batch.reward[:, 0]
        o_t = batch.obs[:, 0]
        s_t = jax.tree.map(lambda x: x[:, 0], batch.state)
        a_t = batch.action[:, 0]
        o_tp1 = batch.obs[:, 1]
        s_tp1 = jax.tree.map(lambda x: x[:, 1], batch.state)
        d_t = batch.done[:, 0]
        tau = temperature
        gamma = discount_factor
        q_network = train_state.apply_fn
        prev_q_t = q_network(train_state.prev_q_network_params, o_t)
        prev_q_tp1 = q_network(train_state.prev_q_network_params, o_tp1)
        target_q_tp1 = q_network(train_state.target_q_network_params, o_tp1)
        pi_t = distrax.Categorical(logits=prev_q_t / tau)
        pi_tp1 = distrax.Categorical(logits=prev_q_tp1 / tau)

        key, step_key = jax.random.split(key)
        r_t = jax.vmap(env.get_reward, in_axes=(0,0,0,None))(s_t, a_t, s_tp1, env_params)

        soft_q_values = jnp.sum(pi_tp1.probs * (target_q_tp1 - tau * pi_tp1.logits), axis=-1)
        target = r_t + alpha * tau * pi_t.log_prob(a_t) + gamma * (1.0 - d_t) * soft_q_values

        key, action_key = jax.random.split(key)
        a_prev_pi_t = pi_t.sample(seed=key)


        def loss_fn(params):
            q_values = q_network(params, o_t)
            q_a_t_values = jnp.take_along_axis(q_values, a_t[:, None], axis=-1).squeeze(1)
            omd_loss = 0.5 * jnp.mean(optax.l2_loss(q_a_t_values, target))
            cql_loss = cql_weight * jnp.mean(nn.logsumexp(q_values, axis=-1) - q_a_t_values)
            return omd_loss + cql_loss

        loss, grad = jax.value_and_grad(loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grad)
        train_state = train_state.replace(
            n_updates=train_state.n_updates + 1
        )

        metrics = {
            "loss": loss
        }
        return train_state, metrics


    def train(key: jax.random.PRNGKey, train_state: OfflineMirrorDescentTrainState, env_params: EnvParams = None) -> tuple[OfflineMirrorDescentTrainState, dict]:
        def _loop_fn(carry, batch):
            key, train_state = carry
            key, train_key = jax.random.split(key)
            train_state, metrics = update(train_key, train_state, batch, env_params)
            return (key, train_state), metrics


        def format_dataset(x):
            y = jnp.stack([x[:, :-1], x[:, 1:]], axis=2)
            y = y.reshape(-1, 2, *x.shape[2:])
            return y
        transition_dataset = jax.tree.map(format_dataset, dataset)

        key, shuffle_key = jax.random.split(key)
        num_transitions = num_batches * batch_size
        indices = jax.random.choice(shuffle_key, jnp.arange(transition_dataset.obs.shape[0]), shape=(num_transitions,), replace=True)
        train_data = jax.tree.map(lambda x: x[indices].reshape(num_batches, batch_size, *x.shape[1:]), transition_dataset)

        train_state = train_state.replace(
            prev_q_network_params=jax.tree.map(jnp.copy, train_state.params),
            target_q_network_params=jax.tree.map(jnp.copy, train_state.params)
        )
        key, train_key = jax.random.split(key)
        (_, train_state), metrics = jax.lax.scan(_loop_fn, (train_key, train_state), train_data)
        train_state = train_state.replace(
            iterations=train_state.iterations + 1
        )
        return train_state, metrics
    
    return NamedTuple("Learner", [
        ("init", Callable[[jax.random.PRNGKey], OfflineMirrorDescentTrainState]),
        ("train", Callable[[jax.random.PRNGKey, OfflineMirrorDescentTrainState, EnvParams], Tuple[OfflineMirrorDescentTrainState, dict]]),
    ])(init, train)

    
