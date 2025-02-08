from .env import Env, Env_Single
from .recorder import Recorder
import gymnasium as gym

gym.register(
    id='MCrafter-single-agent',
    entry_point='mcrafter:Env_Single')

gym.register(
    id='MCrafter-two-agents',
    entry_point='mcrafter:Env')

gym.register(
    id='MCrafter-six-agents',
    entry_point='mcrafter:Env')