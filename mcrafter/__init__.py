from .env import Env, Env_Single
import gymnasium as gym

gym.register(
    id='MCrafter-single-agent',
    entry_point='mcrafter:Env_Single')