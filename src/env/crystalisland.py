import logging
import pickle
import random
from copy import deepcopy

import numpy as np
import src.env.constants as envconst
from gym import Env, spaces

logger = logging.getLogger(__name__)

class CrystalIsland(Env):
    def __init__(self, solution_predictor=None, action_probs=None):
        super(CrystalIsland, self).__init__()

        # based on the defined state, we are creating an observation space
        obs_space_tup = ()
        for fname, fid in envconst.state_map.items():
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
        self.action_space = spaces.Discrete(len(envconst.action_map), )

        self.state = self._get_init_state()
        self.step_count = 0

        self.solution_predictor = solution_predictor
        self.envconst = envconst

        self.action_probs = action_probs

    def _get_init_state(self):
        init_state = np.zeros(len(envconst.state_map))
        init_state[envconst.state_map['s_label_lesson']] = 1

        init_state[envconst.state_map['s_target_disease']] = np.random.choice([0, 1])
        init_state[envconst.state_map['s_target_item']] = np.random.choice([0, 1, 2])
        init_state[envconst.state_map['s_testleft']] = np.random.choice([3, 5, 10])
        for f in [f for f in envconst.state_map.keys() if 's_static_' in f]:
            init_state[envconst.state_map[f]] = np.random.choice([0, 1])
        return deepcopy(init_state)

    def _diff_state(self, state, next_state):
        fchanges = {}
        fids = np.where(state != next_state)[0]
        fvals = next_state[fids]
        for fid, val in zip(fids, fvals):
            fname = envconst.state_map_rev[fid]
            fchanges[fname] = val
        return fchanges

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.state = self._get_init_state()
        diffs = self._diff_state(np.zeros(len(envconst.state_map)), self.state)
        logger.debug("INIT: {0}".format(diffs))
        return self.state

    def set_state(self, state):
        self.state = deepcopy(state)

    def render(self, mode="human"):
        raise NotImplementedError

    def step(self, action: int):
        action_name = envconst.action_map_rev[action]
        next_state = deepcopy(self.state)
        self.step_count += 1
        done = False
        reward = -1.0
        info = {"action_name": action_name}

        rand_val = ""
        aes = ""

        if 'a_talk_' in action_name:
            if action_name == 'a_talk_que' and next_state[envconst.state_map['s' + action_name[1:]]] == 1:
                # talk to quentin 2nd time | AES Quentin Revelation triggered
                next_state, aes = self._trigger_aes(next_state, aes_name='s_aes_que_', action_prob={0: 0.5, 1: 0.5})
            if action_name == 'a_talk_bry' and next_state[envconst.state_map['s' + action_name[1:]]] == 1:
                # talk to bryce 2nd time | AES Bryce Password Revelation triggered
                next_state, aes = self._trigger_aes(next_state, aes_name='s_aes_pas_', action_prob={0: 0.5, 1: 0.5})
            if action_name in ['a_talk_que', 'a_talk_for', 'a_talk_rob'] and \
                    next_state[envconst.state_map['s' + action_name[1:]]] == 0:
                # talk to bryce, ford, or robert for the 1st time | AES Knowledge Quiz triggered
                next_state, aes = self._trigger_aes(next_state, aes_name='s_aes_kno_', action_prob={0: 0.5, 1: 0.5})
            if action_name == 'a_talk_ter':
                # talk to teresa | AES Teresa Symptoms triggered
                next_state, aes = self._trigger_aes(next_state, aes_name='s_aes_ter_', action_prob={0: 0.33, 1: 0.33, 2: 0.34})
            if action_name == 'a_talk_bry':
                # talk to bryce | AES Bryce Symptoms triggered
                next_state, aes = self._trigger_aes(next_state, aes_name='s_aes_bry_', action_prob={0: 0.5, 1: 0.5})

            next_state[envconst.state_map['s' + action_name[1:]]] = 1

        elif 'a_computer' == action_name:
            next_state[envconst.state_map['s' + action_name[1:]]] += 1

        elif 'a_testleft' == action_name:
            next_state[envconst.state_map['s' + action_name[1:]]] += 1

        elif action_name in ['a_notetake', 'a_noteview', 'a_worksheet']:
            next_state[envconst.state_map['s' + action_name[1:]]] += 1

        elif 'a_objtest' == action_name:
            # if we have tests left
            if next_state[envconst.state_map['s_testleft']] > 0:
                next_state[envconst.state_map['s_testleft']] -= 1

                # find items are that were collected and was not tested and randomly choose one
                s_obj = [f for f in envconst.state_map.keys() if ('s_obj_' in f) and
                         (next_state[envconst.state_map[f]] == 1) and
                         (next_state[envconst.state_map['s_objtest_' + f.split('s_obj_')[1]]] == 0)]
                if len(s_obj) > 0:
                    rand_val = np.random.choice(s_obj)
                    s_objtest = 's_objtest_' + rand_val[6:]
                    next_state[envconst.state_map[s_objtest]] = 1

                    # check if target item is the same
                    target_item = next_state[envconst.state_map['s_target_item']]
                    if (rand_val, target_item) in [("s_obj_egg", 0), ("s_obj_mil", 1), ("s_obj_san", 2)]:
                        next_state[envconst.state_map['s_testpos']] = 1
                        next_state[envconst.state_map['s_label_slide']] = 1

        elif action_name in ['a_post', 'a_book', 'a_obj']:
            fnames = [f for f in envconst.state_map.keys() if 's' + action_name[1:] + "_" in f]
            p = None
            if self.action_probs is not None:
                p = [self.action_probs[action_name][f] for f in fnames]
            rand_val = np.random.choice(fnames, p=p)
            next_state[envconst.state_map[rand_val]] = 1

        elif 'a_label' == action_name:
            # labeling must be available | usually lesson is tried much earlier than slide
            if next_state[envconst.state_map["s_label_lesson"]] == 1:
                # on avg 5.2 tries (sd 4.1) is needed for making lesson correct with min 1, max 28
                rand_val = np.random.normal(5.2, scale=4.1)
                if next_state[envconst.state_map["s_label"]] > rand_val:
                    next_state[envconst.state_map["s_label_lesson"]] = 0
                next_state[envconst.state_map["s_label"]] += 1

            elif next_state[envconst.state_map["s_label_slide"]] == 1:
                # on avg 8.4 tries (sd 5.7) is needed (including lesson) for making slide correct with min 1, max 38
                rand_val = np.random.normal(8.4, scale=5.7)
                if next_state[envconst.state_map["s_label"]] > rand_val:
                    next_state[envconst.state_map["s_label_slide"]] = 0
                next_state[envconst.state_map["s_label"]] += 1

        elif 'a_workshsubmit' == action_name:
            next_state[envconst.state_map['s' + action_name[1:]]] += 1
            if self.solution_predictor is not None:
                solved = self.solution_predictor.predict([next_state])[0]
            else:
                # the prob is always around 16%-22% no matter if testpos or lesson read or anything else.
                solved = np.random.choice([0, 1], p=[.8, .2])

            if solved:
                done = True
                reward = 100.0

            # AES Worksheet triggered
            next_state, aes = self._trigger_aes(next_state, aes_name='s_aes_wor_', action_prob={0: 0.33, 1: 0.33, 2: 0.34})

        next_state[envconst.state_map['s_step']] += 1
        # noting down info
        info["rand_val"] = rand_val
        info["aes"] = aes
        info["fchanges"] = self._diff_state(self.state, next_state)

        self.state = next_state
        return next_state, reward, done, info

    def _trigger_aes(self, next_state, aes_name: str, action_prob: dict):
        # action numbers follow Jon's original thesis description respectively
        narr_action = np.random.choice(list(action_prob.keys()), p=list(action_prob.values()))
        aes = aes_name + str(narr_action+1)
        next_state[envconst.state_map[aes]] += 1
        return next_state, aes
