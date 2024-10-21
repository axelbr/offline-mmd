from typing import Any

import chex
import jax
import jax.numpy as jnp
from gymnax.environments.environment import Environment
from gymnax.environments.spaces import Discrete, Box


def get_map(map_str: str) -> jnp.ndarray:
    return jnp.array([[c != "#" for c in line] for line in map_str.strip().split("\n")])


@chex.dataclass
class FourRoomEnvState:
    time: int
    x: int
    y: int


@chex.dataclass
class FourRoomEnvParams:
    mean_field: jnp.ndarray
    time_horizon: int
    initial_state_distribution: jnp.ndarray = None


FOUR_ROOMS = """
#############
#     #     #
#     #     #
#           #
#     #     #
#     #     #
### ##### ###
#     #     #
#     #     #
#           #
#     #     #
#     #     #
#############
"""

LABYRINTH_SMALL = """
#############
#     #     #
#  ####  #  #
#  #     #  #
#  #  ####  #
#     #     #
## ######   #
#  #        #
#  #  #  #  #
#     #  #  #
#  ####  #  #
#     #     #
#############
"""

LABYRINTH = """
######################
#      #     #     # #
#      #     #     # #
######    #  # ##  # #
#         #  # #   # #
#         #  # ### # #
#  ########  #   #   #
#    # # #  ##   #   #
#    # # #     # # ###
#    # # #     # # # #
#  ### # ####### # # #
#  #         #   # # #
#  # ## ###  #   # # #
## # #    #  ##### # #
## # # #  #      # # #
#    # ####        # #
# ####  # ########   #
#       #  #   # ### #
#  #  # #  # # #   # #
# ##### #    # #     #
#            #       #
######################
"""

class FourRoomEnv(Environment):

    def __init__(self, time_horizon: int = 40, task: str = "exploration"):
        '''
        :param map: string representation of the map or a numpy array
        :param time_horizon: maximum number of time steps
        :task: task to be performed in the environment. Either "navigation" or "exploration"
        '''
       
        self.time_horizon = time_horizon
        self.target = None
        if task == "labyrinth":
            self.map = get_map(LABYRINTH)
        else:
            self.map = get_map(FOUR_ROOMS)

        if task == "navigation" or task == "labyrinth":
            self.target = jnp.array([self.map.shape[0] - 2, self.map.shape[1] - 2])

        if task == "labyrinth":
            self.time_horizon = 100

        self.task = task
 
        self.action_map = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]])

    @property
    def default_params(self) -> FourRoomEnvParams:
        initial_state_distribution = jnp.zeros_like(self.map, dtype=jnp.float32)
        initial_state_distribution = initial_state_distribution.at[1, 1].set(1.0)
        initial_state_distribution = initial_state_distribution / initial_state_distribution.sum((0, 1), keepdims=True)
        mf = jnp.repeat(initial_state_distribution[None], self.time_horizon, axis=0)
        return FourRoomEnvParams(mean_field=mf, time_horizon=self.time_horizon, initial_state_distribution=initial_state_distribution)

    def dynamics(self, state, action, next_state):
        x, y = state.x, state.y
        next_x, next_y = next_state.x, next_state.y
        pos = jnp.array([state.x, state.y])
        move = self.action_map[action]
        next_pos = pos + move
        next_pos = jax.lax.select(
            self.map[next_pos[0], next_pos[1]] == 0, pos, next_pos
        )
        return jnp.all(jnp.equal(next_pos, jnp.array([next_x, next_y]))).astype(jnp.float32)

    def step_env(
            self, key: chex.PRNGKey, state: FourRoomEnvState, action: int, params: FourRoomEnvParams
    ) -> tuple[chex.Array, FourRoomEnvState, jnp.ndarray, jnp.ndarray, dict[Any, Any]]:
        mean_field = params.mean_field
        move = self.action_map[action]
        pos = jnp.array([state.x, state.y])
        next_pos = pos + move
        next_pos = jax.lax.select(
            self.map[next_pos[0], next_pos[1]] == 0, pos, next_pos
        )
        next_state = FourRoomEnvState(time=state.time + 1, x=next_pos[0], y=next_pos[1])
        obs = self.get_obs(next_state, params=params, key=key)
        reward = self.get_reward(state, action, next_state, params)
        done = self.is_terminal(next_state, params)
        return obs, next_state, reward, done, {}

    def reset_env(self, key: chex.PRNGKey, params: FourRoomEnvParams) -> tuple[chex.Array, FourRoomEnvState]:
        key, init_key = jax.random.split(key)
        dist = jnp.where(self.map, params.initial_state_distribution, 0)
        dist = dist / dist.sum()
        idx = jax.random.categorical(init_key, jnp.log(dist.reshape(-1)))
        x, y = jnp.unravel_index(idx, params.initial_state_distribution.shape)
        state = FourRoomEnvState(time=0, x=x, y=y)
        obs = self.get_obs(state, params=params, key=key)
        return obs, state

    def get_obs(self, state, params=None, key=None) -> chex.Array:
        max_x, max_y = self.map.shape
        obs = jnp.zeros(1 + max_x + max_y)
        t = state.time / params.time_horizon
        obs = obs.at[state.x].set(1)
        obs = obs.at[max_x + state.y].set(1)
        obs = obs.at[-1].set(t)
        return obs

    def get_reward(self, state: FourRoomEnvState, action: int, next_state: FourRoomEnvState, params: FourRoomEnvParams) -> jnp.ndarray:
        state_density = params.mean_field[state.time, state.x, state.y]

        if self.task == "exploration":
            return -jnp.log(jnp.clip(state_density, 1e-6, 1.0))  
        elif self.task == "navigation":
            dist = jnp.abs(state.x - self.target[0]) + jnp.abs(state.y - self.target[1])
            max_dist = jnp.abs(self.target[0] - 1) + jnp.abs(self.target[1] - 1)
            congestion = state_density * jnp.sum(jnp.abs(self.action_map[action]))
            crowding_reward = -jnp.log(jnp.clip(state_density, 1e-6, 1.0))
            reward = crowding_reward - 5 * dist - congestion
            return reward
            #reward = reward - dist - 10 * congestion
    
            #jax.debug.print("reward {}", reward)
            #jax.debug.print("congestion {}", congestion)
            #jax.debug.print("target_reached {}", target_reached)

        elif self.task == "labyrinth":
            dist = jnp.abs(state.x - self.target[0]) + jnp.abs(state.y - self.target[1])
            max_dist = jnp.abs(self.target[0] - 1) + jnp.abs(self.target[1] - 1)
            congestion = state_density * jnp.sum(jnp.abs(self.action_map[action]))
            crowding_reward = -jnp.log(jnp.clip(state_density, 1e-6, 1.0)) / -jnp.log(1e-6)
            return -1 * dist / max_dist + crowding_reward - congestion
        else:
            raise ValueError("Invalid task")
        
    def get_env_state(self, t: int, index: int) -> FourRoomEnvState:
        x = index // self.map.shape[1]
        y = index % self.map.shape[1]
        return FourRoomEnvState(time=t, x=x, y=y)


    def is_terminal(self, state: FourRoomEnvState, params: FourRoomEnvParams) -> jnp.ndarray:
        return state.time == params.time_horizon

    def action_space(self, params: FourRoomEnvParams):
        return Discrete(len(self.action_map))

    def observation_space(self, params: FourRoomEnvParams):
        max_x, max_y = self.map.shape
        return Box(
            low=0,
            high=1,
            shape=(1 + max_x + max_y,),
            dtype=jnp.float32
        )

    def state_space(self, params: FourRoomEnvParams):
        return Discrete(params.mean_field.shape[1] * params.mean_field.shape[2])

