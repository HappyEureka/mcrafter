import argparse

import numpy as np
try:
  import pygame
except ImportError:
  print('Please install the pygame package to use the GUI.')
  raise
from PIL import Image

import crafter


def main():
  boolean = lambda x: bool(['False', 'True'].index(x))
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=None)
  parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
  parser.add_argument('--view', type=int, nargs=2, default=(9, 9))
  parser.add_argument('--length', type=int, default=None)
  parser.add_argument('--health', type=int, default=9)
  parser.add_argument('--window', type=int, nargs=2, default=(600, 600))
  parser.add_argument('--size', type=int, nargs=2, default=(0, 0))
  parser.add_argument('--record', type=str, default=None)
  parser.add_argument('--fps', type=int, default=5)
  parser.add_argument('--wait', type=boolean, default=False)
  parser.add_argument('--death', type=str, default='reset', choices=[
      'continue', 'reset', 'quit'])
  parser.add_argument('--n_players', type=int, default=3)
  
  args = parser.parse_args()

  keymap = {
      pygame.K_a: 'move_left',
      pygame.K_d: 'move_right',
      pygame.K_w: 'move_up',
      pygame.K_s: 'move_down',
      pygame.K_SPACE: 'do',
      pygame.K_TAB: 'sleep',

      pygame.K_r: 'place_stone',
      pygame.K_t: 'place_table',
      pygame.K_f: 'place_furnace',
      pygame.K_p: 'place_plant',

      pygame.K_1: 'make_wood_pickaxe',
      pygame.K_2: 'make_stone_pickaxe',
      pygame.K_3: 'make_iron_pickaxe',
      pygame.K_4: 'make_wood_sword',
      pygame.K_5: 'make_stone_sword',
      pygame.K_6: 'make_iron_sword',
      
      # multi player
      pygame.K_LSHIFT: 'switch_player',
      pygame.K_SEMICOLON: 'pause',
  }
  
  print('Actions:')
  for key, action in keymap.items():
    print(f'  {pygame.key.name(key)}: {action}')

  crafter.constants.items['health']['max'] = args.health
  crafter.constants.items['health']['initial'] = args.health

  size = list(args.size)
  size[0] = size[0] or args.window[0]
  size[1] = size[1] or args.window[1]

  env = crafter.Env(
      area=args.area, view=args.view, length=args.length, seed=args.seed, n_players=args.n_players)
  env = crafter.Recorder(env, args.record)
  env.reset()
  achievements = set()
  duration = 0
  return_ = 0
  was_done = False
  print('Diamonds exist:', env._world.count('diamond'))

  pygame.init()
  screen = pygame.display.set_mode(args.window)
  clock = pygame.time.Clock()
  running = True
  pid=0
  while running:

    # Rendering.
    image = env.render(size)
    if size != args.window:
      image = Image.fromarray(image)
      image = image.resize(args.window, resample=Image.NEAREST)
      image = np.array(image)
    surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(args.fps)

    # Keyboard input.
    action = None
    pygame.event.pump()
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        running = False
      elif event.type == pygame.KEYDOWN and event.key in keymap.keys():
        action = keymap[event.key]
    if action is None:
      pressed = pygame.key.get_pressed()
      for key, action in keymap.items():
        if pressed[key]:
          break
      else:
        # added for multi player
        if args.wait and not (env._player.sleeping or env._player2.sleeping):
          continue
        
        else:
          action = 'noop'

    print('Action:', action, env.player_id)
    if action == 'switch_player':
      pid += 1
      pid %= env.n_players
      env.switch_player(pid)
      continue
    elif action == 'pause':
      auxiliary_action = ""
      while auxiliary_action not in ['e', 'c', 'exit']:
        auxiliary_action = input("Please select one of the following actions [exchange item (e), communicate (c)]: ")
      if auxiliary_action == 'e':
        target_player_id = -1
        print(f"You are Player {env.player_id}.")
        print("Here is your inventory:", env._players[env.player_id].inventory)
        print(f"There are {env.n_players} players in total and their ids are 0~{env.n_players-1}. The alive players are {env._alive_players_id}.")
        target_player_id = int(input(f"Please select a player (id) to exchange item: "))
        item = input("Please select an item to exchange: ")
        env.exchange_item(target_player_id, item)
      elif auxiliary_action == 'c':
        def _change_target_chat_player_ids():
          print("Chatting mode is on. You can type 'exit' to exit chatting mode; 'history' to see past interactions; 'change' to change the target player ids.")
          print(f"There are {env.n_players} players in total and their ids are 0~{env.n_players-1}. The alive players are {env._alive_players_id}.")
          target_player_ids = input(f"Please select the player ids you want to communicate with (separate by space) or all (a): ")
          if target_player_ids == 'a':
            target_player_ids = list(range(env.n_players))
          else:
            target_player_ids = target_player_ids.split()
            target_player_ids = [int(id_) for id_ in target_player_ids]
          target_player_ids.append(env.player_id)
          return set(target_player_ids)
          
        chat = ""
        target_player_ids = _change_target_chat_player_ids()
        while chat != 'exit':
          if chat == 'history':
            env.show_history()
            chat = ""
          elif chat == 'change':
            target_player_ids = _change_target_chat_player_ids()
          elif chat != "":
            msg = {'role': f"Player {env.player_id}", 'message': chat, "to": target_player_ids}
            env.chat(msg, target_player_ids)
          chat = input(f"Player {env.player_id}: ")
      elif auxiliary_action == 'exit':
        continue
      continue
      
    # Environment step.
    actions = ['noop'] * env.n_players
    actions[env.player_id] = action
    actions_index_list = [env.action_names.index(action) for action in actions]
    _, reward, done, _ = env.step(actions_index_list)
    duration += 1

    # Achievements.
    # unlocked = {
    #     name for name, count in env._player.achievements.items()
    #     if count > 0 and name not in achievements}
    # for name in unlocked:
    #   achievements |= unlocked
    #   total = len(env._player.achievements.keys())
    #   print(f'Achievement ({len(achievements)}/{total}): {name}')
    # if env._step > 0 and env._step % 100 == 0:
    #   print(f'Time step: {env._step}')
    # if reward:
    #   print(f'Reward: {reward}')
    #   return_ += reward

    # Episode end.
    if done and not was_done:
      was_done = True
      print('Episode done!')
      print('Duration:', duration)
      print('Return:', return_)
      if args.death == 'quit':
        running = False
      if args.death == 'reset':
        print('\nStarting a new episode.')
        env.reset()
        achievements = set()
        was_done = False
        duration = 0
        return_ = 0
      if args.death == 'continue':
        pass

  pygame.quit()


if __name__ == '__main__':
  main()
