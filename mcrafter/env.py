import collections
import numpy as np
from . import constants
from . import engine
from . import objects
from . import worldgen
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
from gymnasium import Env as GymnasiumEnv
import functools

NONE_EXCHANGE_ITEMS = ['health', 'energy']

class Env_Single(GymnasiumEnv):
    def __init__(self):
        """
        Wraps a PettingZoo parallel environment to make it work as a single-agent Gymnasium environment.

        Args:
            parallel_env: The PettingZoo ParallelEnv to wrap.
            agent_id: The ID of the agent to focus on in single-agent mode.
        """
        super().__init__()
        self.parallel_env = Env(n_players=1, render_mode='human')
        self.action_space = self.parallel_env._action_space
        self.observation_space = self.parallel_env._observation_space
        self.metadata = self.parallel_env.metadata
    
    # def observation_space(self, agent=None):
    #     return self._observation_space

    # def action_space(self, agent=0):
    #     return self._action_space
    
    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        return self.parallel_env.reset()

    def step(self, action):
        """Executes a step for the specified agent in the environment."""
        return self.parallel_env.step({'0': action})

    def render(self, mode="human"):
        """Render the environment (delegates to the parallel environment's render method)."""
        return self.parallel_env.render(mode)

    def close(self):
        """Closes the environment (delegates to the parallel environment's close method)."""
        self.parallel_env.close()
        
        
        
class Env(ParallelEnv):
    def __init__(
        self, area=(64, 64), view=(9, 9), size=(64, 64),
        reward=True, length=10000, n_players=1, seed=42, render_mode='human'
    ):
        # Initialize environment attributes
        view = np.array(view if hasattr(view, '__len__') else (view, view))
        size = np.array(size if hasattr(size, '__len__') else (size, size))
        seed = np.random.randint(0, 2**31 - 1) if seed is None else seed
        self.render_mode = render_mode
        self._area = area
        self._view = view
        self._size = size
        self._reward = reward
        self._length = length
        self._seed = seed
        self._episode = 0
        self._world = engine.World(area, constants.materials, (12, 12))
        self._textures = engine.Textures(constants.root / 'assets')
        item_rows = int(np.ceil(len(constants.items) / view[0]))
        self._local_view = engine.LocalView(
            self._world, self._textures, [view[0], view[1] - item_rows])
        self._item_view = engine.ItemView(
            self._textures, [view[0], item_rows])
        self._sem_view = engine.SemanticView(self._world, [
            objects.Player, objects.Cow, objects.Zombie,
            objects.Skeleton, objects.Arrow, objects.Plant])
        self._step = None
        self._unlocked = {}

        self._observation_space = Box(0, 255, tuple(self._size) + (3,), np.uint8)
        self._action_space = Discrete(len(constants.actions))
        #self._action_space = Discrete(1+4+1+1+1) # only allow noop, moves, do, place_table, make_wood_pickaxe
        self.player_id = 0
        self.n_players = n_players
        self.agents = []
        self.canvases = []
        self._players = [None] * self.n_players

        # Convert agent IDs to strings
        self.possible_agents = [str(i) for i in range(self.n_players)]
        self._last_healths = [None] * self.n_players
        self._last_inventory = [None] * self.n_players

        self.reward_range = None
        self.metadata = {"is_parallelizable": True}

    def switch_player(self, player_id):
        self.player_id = player_id

    def exchange_item(self, target_player_id, item):
        if item in NONE_EXCHANGE_ITEMS:
            print("Cannot exchange item: ", item)
            return
        if self.player_id == target_player_id:
            print("Cannot exchange item with self.")
            return
        curr_player = self._players[self.player_id]
        if curr_player.inventory[item] > 0:
            curr_player.inventory[item] -= 1
            self._players[target_player_id].inventory[item] += 1

    def observation_space(self, agent=None):
        return self._observation_space

    def action_space(self, agent=0):
        return self._action_space

    @property
    def action_names(self):
        return constants.actions

    def reset(self, seed=42, options=None):
        """Reset the environment."""
        self._episode += 1
        self._step = 0
        self.agents = self.possible_agents[:]  # Reset agents list to all agents

        self._world.reset(seed=42)
        # if seed is not None:
        #     self._world.reset(seed=seed)
        # else:
        #     print("seed!")
        #     self._world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
        self._update_time()

        center = (self._world.area[0] // 2, self._world.area[1] // 2)
        x, y = center

        for i, agent_id in enumerate(self.possible_agents):
            self._players[i] = objects.Player(self._world, (x + i - self.n_players // 2, y))
            self._last_healths[i] = self._players[i].health
            self._last_inventory[i] = self._players[i].inventory.copy()
            self._world.add(self._players[i])

        self._unlocked = set()
        worldgen.generate_world(self._world, self._players[self.n_players // 2])
        self.canvases = []
        self.render_all()

        infos = {agent_id: {} for agent_id in self.possible_agents}
        observations = self._obs()

        if self.n_players == 1:
            return observations[self.possible_agents[0]], infos[self.possible_agents[0]]
        return observations, infos

    def step(self, actions):
        """Step the environment forward."""
        self._step += 1
        self._update_time()

        player_rewards = {}
        unlocked = set()
        players_to_remove = []

        for agent_id in self.agents:
            action = actions[agent_id]
            curr_reward, is_alive = self.step_one_player(action, int(agent_id))  # Convert back to int for internal logic
            player_rewards[agent_id] = curr_reward
            if not is_alive:
                players_to_remove.append(agent_id)

        return_terminated = {agent_id: self._players[int(agent_id)].achievements['collect_diamond'] > 0 for agent_id in self.agents}
        return_truncated = {agent_id: False for agent_id in self.agents}

        if len(players_to_remove) > 0:
            for agent_id in players_to_remove:
                self.agents.remove(agent_id)
                return_truncated[agent_id] = True

        for obj in self._world.objects:
            obj.update()
        if self._step % 10 == 0:
            for chunk, objs in self._world.chunks.items():
                self._balance_chunk(chunk, objs)

        self.render_all()
        obs = self._obs()

        return_obs = {agent_id: obs[agent_id] for agent_id in self.agents + players_to_remove}
        return_reward = {agent_id: player_rewards[agent_id] for agent_id in self.agents + players_to_remove}
        return_info = {agent_id: {} for agent_id in self.agents + players_to_remove}
        
        if self.n_players == 1:
            return return_obs[self.possible_agents[0]], return_reward[self.possible_agents[0]], return_terminated[self.possible_agents[0]], return_truncated[self.possible_agents[0]], return_info[self.possible_agents[0]]
        return return_obs, return_reward, return_terminated, return_truncated, return_info
    
    # def step(self, actions):
    #     self._step += 1
    #     self._update_time()

    #     player_rewards = {}
    #     unlocked = set()
    #     players_to_remove = []
    #     for player_id in self.agents:
    #         action = actions[player_id]
    #         curr_reward, is_alive = self.step_one_player(action, player_id)
    #         player_rewards[player_id] = curr_reward
    #         if not is_alive:
    #             players_to_remove.append(player_id)

    #     return_terminated = {alive_agent_id: self._players[alive_agent_id].achievements['collect_diamond'] > 0 for alive_agent_id in self.agents}
    #     return_truncated = {alive_agent_id: False for alive_agent_id in self.agents}
        
    #     if len(players_to_remove) > 0:
    #         for player_id in players_to_remove:
    #             self.agents.remove(player_id)
    #             return_truncated[player_id] = True

    #     for obj in self._world.objects:
    #         obj.update()
    #     if self._step % 10 == 0:
    #         for chunk, objs in self._world.chunks.items():
    #             self._balance_chunk(chunk, objs)

    #     self.render_all()
    #     obs = self._obs()

    #     info = []
    #     reward = []
    #     for id_, p in enumerate(self._players):
    #         reward.append(player_rewards.get(id_, 0))
    #         info_p = {
    #             'id': id_,
    #             'inventory': p.inventory.copy(),
    #             'achievements': p.achievements.copy(),
    #             'sleeping': p.sleeping,
    #             'discount': 1 - float(p.health <= 0),
    #             'semantic': self._sem_view(),
    #             'player_pos': p.pos,
    #             'player_facing': p.facing,
    #             'reward': reward[-1],
    #             'dead': p.health <= 0,
    #             'unlocked': unlocked,
    #             'action': p.action,
    #             'view': self._view,
    #         }
    #         info.append(info_p)
            
    #     return_obs = {alive_agent_id: obs[alive_agent_id] for alive_agent_id in self.agents + players_to_remove}
    #     return_reward = {alive_agent_id: reward[alive_agent_id] for alive_agent_id in self.agents + players_to_remove}
    #     return_info = {alive_agent_id: info[alive_agent_id] for alive_agent_id in self.agents + players_to_remove}
    #     if self.n_players == 1:
    #         return return_obs[0], return_reward[0], return_terminated[0], return_truncated[0], return_info[0]
    #     return return_obs, return_reward, return_terminated, return_truncated, return_info
        

    def step_one_player(self, action, player_id):
        # print(constants.actions, action, constants.actions[action]) #
        """
        ['noop', 'move_left', 'move_right', 'move_up', 'move_down', 'do', 'sleep', 'place_stone', 
        'place_table', 'place_furnace', 'place_plant', 'make_wood_pickaxe', 'make_stone_pickaxe', 
        'make_iron_pickaxe', 'make_wood_sword', 'make_stone_sword', 'make_iron_sword', 'share'] 
        3 
        move_up
        
        from 0-5 are good. map 6 to place table, 7 to make wood pickaxe
        """
        # if action == 6:
        #     action = 8
        # elif action == 7:
        #     action = 11
        
        curr_player = self._players[player_id]
        curr_player.action = constants.actions[action]

        curr_player_reward = 0 #(curr_player.health - self._last_healths[player_id])
        self._last_healths[player_id] = curr_player.health

        task_difficulties = {
            'collect_wood': 5, 'collect_sapling': 1, 'place_plant': 0, 'eat_plant': 0, 'collect_stone': 50,
            'make_wood_pickaxe': 20, 'make_wood_sword': 0, 'collect_coal': 50, 'collect_iron': 50,
            'make_stone_pickaxe': 30, 'make_stone_sword': 0, 'place_table': 10, 'place_furnace': 5,
            'collect_drink': 1, 'wake_up': 0, 'make_iron_pickaxe': 7, 'make_iron_sword': 0,
            'eat_cow': 1, 'collect_diamond': 10, 'place_stone': 1, 'defeat_zombie': 0, 'defeat_skeleton': 0,
            'wood': 5, 'stone': 20, 'coal': 20, 'iron': 50, 'diamond': 100,
        }

        unlocked = {
            name for name, count in curr_player.achievements.items()
            if count > 0 and name not in self._unlocked
        }

        
        if action == 0:
            curr_player_reward += 0
        elif action < 5:
            curr_player_reward += 3
        else:
            curr_player_reward += 1

        if unlocked:
            self._unlocked |= unlocked
            for t in unlocked:
                curr_player_reward += task_difficulties[t] * 10
            
            #print(self._unlocked, curr_player_reward)

        diff_inventory = {k: curr_player.inventory[k] - self._last_inventory[player_id][k] for k in curr_player.inventory if curr_player.inventory[k] != self._last_inventory[player_id][k]}
        for k, v in diff_inventory.items():
            if k in task_difficulties and v > 0:
                curr_player_reward += task_difficulties[k] * 10
        self._last_inventory[player_id] = curr_player.inventory.copy()

        is_alive = True
        if curr_player.health <= 0:
            self._world.remove(curr_player)
            is_alive = False
        return curr_player_reward, is_alive

    def render_one_player(self, player_id, size=None):
        size = size or self._size
        unit = size // self._view
        canvas = np.zeros(tuple(size) + (3,), np.uint8)

        curr_player = self._players[player_id]
        local_view = self._local_view(curr_player, unit)
        item_view = self._item_view(curr_player.inventory, unit)
        view = np.concatenate([local_view, item_view], 1)

        border = (size - (size // self._view) * self._view) // 2
        (x, y), (w, h) = border, view.shape[:2]
        canvas[x: x + w, y: y + h] = view
        return canvas.transpose((1, 0, 2))

    def render_all(self, size=None):
        self.canvases = []
        for player_id in range(len(self._players)):
            self.canvases.append(self.render_one_player(player_id, size))
        return self.canvases

    def render(self, size=None):
        self.render_all(size)
        return self.canvases[self.player_id]

    # def _obs(self):
    #     return {a: canvas for a, canvas in enumerate(self.canvases)}
    def _obs(self):
        """Return observations as a dictionary with string keys."""
        return {str(a): canvas for a, canvas in enumerate(self.canvases)}

    def _update_time(self):
        progress = (self._step / 300) % 1 + 0.3
        daylight = 1 - np.abs(np.cos(np.pi * progress)) ** 3
        self._world.daylight = daylight

    def _balance_chunk(self, chunk, objs):
        light = self._world.daylight
        self._balance_object(
            chunk, objs, objects.Cow, 'grass', 5, 5, 0.01, 0.1,
            lambda pos: objects.Cow(self._world, pos),
            lambda num, space: (0 if space < 30 else 1, 1.5 + light))

    def _balance_object(
        self, chunk, objs, cls, material, span_dist, despan_dist,
        spawn_prob, despawn_prob, ctor, target_fn):

        xmin, xmax, ymin, ymax = chunk
        random = self._world.random
        creatures = [obj for obj in objs if isinstance(obj, cls)]
        mask = self._world.mask(*chunk, material)
        target_min, target_max = target_fn(len(creatures), mask.sum())
        if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
            xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
            ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
            xs, ys = xs[mask], ys[mask]
            i = random.randint(0, len(xs))
            pos = np.array((xs[i], ys[i]))
            empty = self._world[pos][1] is None
            away = True
            for p in self._players:
                _away = p.distance(pos) >= span_dist
                away = away and _away
            if empty and away:
                self._world.add(ctor(pos))
        elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
            obj = creatures[random.randint(0, len(creatures))]
            away = True
            for p in self._players:
                _away = p.distance(obj.pos) >= despan_dist
                away = away and _away
            if away:
                self._world.remove(obj)
