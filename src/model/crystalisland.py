import dataclasses
import logging
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _gen_narrative_action(narrative_state: list, random_planner: bool) -> (int, int, int):
    """
    helper function that generates a narrative action along with AES action and AES state position. when random
    is given, it generates uniform random narrative planner action based on the AES that was triggered.
    :param narrative_state: list that contains the narrative state, that can be used in the narrative planner to
    learn the next action. for random, we only look at the last 4 positions [26, 29] to know which AES was
    triggered.
    :return: a tuple that contains original narrative action, corresponding AES action with AES state position
    """
    narrative_action = -1
    aes_state_num = -1
    aes_action = -1
    if random_planner:
        if narrative_state[26] == 1:  # bryce trigger
            narrative_action = np.random.choice([0, 1])
            aes_state_num = 19
            aes_action = narrative_action + 1
        elif narrative_state[27] == 1:  # teresa trigger
            narrative_action = np.random.choice([2, 3, 4])
            aes_state_num = 20
            aes_action = narrative_action - 1
        elif narrative_state[28] == 1:  # quiz trigger
            narrative_action = np.random.choice([5, 6])
            aes_state_num = 21
            aes_action = narrative_action - 4
        elif narrative_state[29] == 1:  # diagnosis trigger
            narrative_action = np.random.choice([7, 8, 9])
            aes_state_num = 22
            aes_action = narrative_action - 6
        else:
            logger.error("narrative planner state do not have any trigger!")
    else:
        # TODO WHEN WE HAVE AN AGENT
        logger.error("TODO WHEN WE HAVE AN AGENT")

    return narrative_action, aes_action, aes_state_num


def _set_student_state(state: list, aes_action: int, aes_state_num: int) -> list:
    """
    helper function that create student state from narrative actions converted to AES actions. note, AES action
    ranges from [1-3] or [1-2] for each AES but narrative action ranges from [0-9] that includes all AES.
    :param aes_action: for each AES, a value between [1-3] or [1-2]. for example, for teresa's symptom, 1 is
    minimum detail, 2 is moderate detail, 3 is maximum detail.
    :param aes_state_num: this is the AES feature position in the student state. for example, teresa's symptom's
    feature position is 20.
    :return: a list containing the student state
    """
    # resetting last AES
    state[19] = 0
    state[20] = 0
    state[21] = 0
    state[22] = 0

    state[aes_state_num] = aes_action

    return state


def _gen_narrative_state(state, student_action: int) -> list:
    """
    helper function that generates narrative state from the given state and student action. note that,
    narrative state are extension of student state where position [0-25] is the same and [26-29] are AES triggers
    :param student_action: the last action that was taken by the student
    :return: narrative state of size 30
    """
    narrative_state = deepcopy(state)
    narrative_state = np.concatenate([narrative_state, np.array([0, 0, 0, 0])])

    if student_action == 14:  # quiz trigger
        narrative_state[28] = 1
    elif student_action == 15:  # bryce trigger
        narrative_state[26] = 1
    elif student_action == 17:  # teresa trigger
        narrative_state[27] = 1
    elif student_action == 18:  # diagnosis trigger
        narrative_state[29] = 1

    return narrative_state


class CrystalIsland:
    def __init__(self, args: dataclasses, s0: np.ndarray):
        self.args = args
        self.s0 = s0
        self.state = self.s0[np.random.choice(self.s0.shape[0], 1, replace=False), :][0]
        self.is_random_planner = self.args.is_random_planner

        # TODO load narrative planner
        self.narrative_planner = None

    def get_student_state(self):
        return deepcopy(self.state)

    def reset(self):
        self.state = self.s0[np.random.choice(self.s0.shape[0], 1, replace=False), :][0]
        return deepcopy(self.state)

    def step(self, action: int, dryrun: bool = False) -> (list, float, bool, dict):
        state, reward, done, info = self.simulate_step(action, deepcopy(self.state))
        if dryrun is False:
            self.state = state
        return state, reward, done, info

    def simulate_step(self, action: int, state: list) -> (list, float, bool, dict):
        """
        given a action between 0-18, it generates the next state accordingly. if end game (5) or AES (14, 15, 17, 18)
        are triggered, then necessary AES states are also produced according to the narrative planner.
        TODO no reward is implemented
        :param state: the student state from where the action will be taken
        :param action: must be a valid student action [0, 18]
        :return: a tuple with state, reward, done, and info
        """
        # action number and state position are matched for all student actions
        state[action] += 1
        done = False
        reward = 0
        info = {}
        if action == 5:  # s_end_game
            done = True
        elif action in [14, 15, 17, 18]:  # AES trigger
            narrative_state = _gen_narrative_state(state=state, student_action=action)
            narrative_action, aes_action, aes_state_num = _gen_narrative_action(
                narrative_state=narrative_state, random_planner=self.is_random_planner)
            state = _set_student_state(state=state, aes_action=aes_action, aes_state_num=aes_state_num)

            info = {'narrative_state': narrative_state, 'narrative_action': narrative_action}

            logger.debug("for student action {0}, narrative planner action {1} converted to aes action {2} with aes "
                         "position {3} was triggered".format(action, narrative_action, aes_action, aes_state_num))

        return state, reward, done, info

    def gen_random_data(self, steps):
        ep = 0
        data = []
        step = 0
        while step < steps:
            state = deepcopy(self.reset())
            ep_step = 0
            while ep_step < self.args.max_episode_len and step < steps:
                action = np.random.choice(range(0, self.args.action_dim))
                next_state, reward, done, info = self.step(action)
                data.append({'student_id': str(ep), 'step': ep_step, 'state': state, 'action': action, 'reward': reward,
                             'done': done, 'info': info})
                ep_step += 1
                state = deepcopy(next_state)
                step += 1
                if done:
                    break
            ep += 1
            if (step+1) % 10000 == 0:
                logger.info("{0} out of {1} random data generated".format(step+1, steps))

        df = pd.DataFrame(data, columns=['student_id', 'step', 'state', 'action', 'reward', 'done', 'info'])
        return df


