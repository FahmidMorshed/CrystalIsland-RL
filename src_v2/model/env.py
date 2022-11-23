from copy import deepcopy

import numpy as np
import gym
import random

import pandas as pd
from gym import Env, spaces
import pickle

class CrystalIsland(Env):
    def __init__(self, loc_constants="../processed_data/constants.pkl"):
        super((CrystalIsland, self)).__init__()

        self.constants = pickle.load(open(loc_constants, 'rb'))

        # based on the defined state, we are creating an observation space
        obs_space_tup = ()
        for fname, fid in self.constants['state_map'].items():
            if fname in ['s_testleft', 's_label', 's_worksheet', 's_workshsubmit', 's_notetake', 's_noteview',
                         's_computer',
                         's_quiz'] or 's_aes_' in fname:
                obs_space_tup += (spaces.Discrete(200),)  # max count
            elif fname == 's_target_item':
                obs_space_tup += (spaces.Discrete(3),)
            else:
                obs_space_tup += (spaces.Discrete(2),)
        self.observation_space = spaces.Tuple(obs_space_tup)

        # based on the defined action, we are creating an observation space
        self.action_space = spaces.Discrete(len(self.constants['action_map']), )

        self.state = self.reset()

    def reset(self) -> np.ndarray:
        init_state = np.zeros(len(self.constants['state_map']))
        init_state[self.constants['state_map']['s_label_lesson']] = 1
        for f in [f for f in self.constants['state_map'].keys() if 's_static_' in f]:
            init_state[self.constants['state_map'][f]] = np.random.choice([0, 1])
        return deepcopy(init_state)

    def render(self, mode="human"):
        raise NotImplementedError

    def _set_loc(self, state: np.ndarray, s_loc: str) -> np.ndarray:
        # reset all location features and set to current one
        for f in [f for f in self.constants['state_map'].keys() if 's_loc_' in f]:
            state[self.constants['state_map'][f]] = 0
        state[self.constants['state_map'][s_loc]] = 1
        return state

    def _get_loc(self, state: np.ndarray) -> str:
        # reset all location features and set to current one
        for f in [f for f in self.constants['state_map'].keys() if 's_loc_' in f]:
            if state[self.constants['state_map'][f]] == 1:
                return f
        raise NotImplementedError

    def step(self, action: int):
        action_name = self.constants['action_map_rev'][action]
        next_state = deepcopy(self.state)

        if 'a_loc_' in action_name:
            s_loc = 's' + action_name[1:]
            next_state = self._set_loc(next_state, s_loc)

        elif 'a_talk_' in action_name:
            s_loc = self.constants['talk_loc_map']['s' + action_name[1:]]
            next_state = self._set_loc(next_state, s_loc)
            next_state[self.constants['state_map']['s' + action_name[1:]]] = 1

        elif 'a_workshsubmit' == action_name:
            s_loc = 's_loc_kim'
            next_state = self._set_loc(next_state, s_loc)
            next_state[self.constants['state_map']['s' + action_name[1:]]] += 1
            # TODO GAME END

        elif 'a_post' == action_name:
            s_post = np.random.choice(list(self.constants['post_loc_map'].keys()))
            s_loc = self.constants['post_loc_map'][s_post]
            next_state = self._set_loc(next_state, s_loc)
            next_state[self.constants['state_map'][s_post]] = 1

        elif 'a_book' == action_name:
            s_book = np.random.choice([f for f in self.constants['state_map'].keys() if 's_book_' in f])
            s_loc = 's_loc_lab'
            next_state = self._set_loc(next_state, s_loc)
            next_state[self.constants['state_map'][s_book]] = 1

        elif 'a_obj' == action_name:
            s_obj = np.random.choice(list(self.constants['obj_loc_map'].keys()))
            s_loc = self.constants['obj_loc_map'][s_obj]
            next_state = self._set_loc(next_state, s_loc)
            next_state[self.constants['state_map'][s_obj]] = 1

        elif 'a_computer' == action_name:
            s_loc = 's_loc_bry'
            next_state = self._set_loc(next_state, s_loc)
            next_state[self.constants['state_map']['s' + action_name[1:]]] += 1

        elif 'a_testleft' == action_name:
            next_state[self.constants['state_map']['s_quiz']] += np.random.choice([2, 3, 4, 5])
            next_state[self.constants['state_map']['s' + action_name[1:]]] += 1

        elif action_name in ['a_notetake', 'a_noteview', 'a_worksheet']:
            next_state[self.constants['state_map']['s' + action_name[1:]]] += 1

        elif 'a_objtest' == action_name:
            s_loc = 's_loc_lab'
            next_state = self._set_loc(next_state, s_loc)
            # if we have tests left
            if next_state[self.constants['state_map']['s_testleft']] > 0:
                next_state[self.constants['state_map']['s_testleft']] -= 1

                # find a random item to test
                s_obj = np.random.choice(list(self.constants['obj_loc_map'].keys()))
                s_objtest = 's_objtest_' + s_obj[6:]
                next_state[self.constants['state_map'][s_objtest]] = 1

                # check if target item is the same
                target_item = next_state[self.constants['state_map']['s_target_item']]
                if (s_obj, target_item) in [("s_obj_egg", 0), ("s_obj_mil", 1), ("s_obj_san", 2)]:
                    next_state[self.constants['state_map']['s_testpos']] = 1
                    next_state[self.constants['state_map']['s_label_slide']] = 1

        elif 'a_label' == action_name:
            # TODO get a sense of how many label actions are needed to get a success for lesson and slide individually
            # labeling must be available | usually lesson is tried much earlier
            if next_state[self.constants['state_map']["s_label_lesson"]] == 1:
                next_state[self.constants['state_map']["s_label"]] += 1
                solved = np.random.choice([True, False])
                # if solved, no longer available
                if solved:
                    next_state[self.constants['state_map']["s_label_lesson"]] = 0
                    
            elif next_state[self.constants['state_map']["s_label_slide"]] == 1:
                next_state[self.constants['state_map']["s_label"]] += 1
                solved = np.random.choice([True, False])
                # if solved, no longer available
                if solved:
                    next_state[self.constants['state_map']["s_label_slide"]] = 0

        done = False

