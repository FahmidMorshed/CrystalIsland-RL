import dataclasses
import logging
import pickle
from copy import deepcopy

import numpy as np

logger = logging.getLogger(__name__)


class CrystalIsland:
    def __init__(self, args: dataclasses):
        self.args = args
        self.state = [0] * self.args.state_dim

        # TODO load narrative planner
        self.narrative_planner = None

    def take_student_step(self, action):
        # action number and state position are matched for all student actions
        action = int(action)
        self.state[action] += 1
        done = False
        reward = 0

        if action == 5: # s_end_game
            done = True
        elif action in [14, 15, 17, 18]: # AES trigger
            narrative_state = self._gen_narrative_state(student_action=action)
            narrative_action, aes_state_num, aes_action = self._gen_narrative_action(narrative_state)
            self.state = self._set_student_state(aes_action, aes_state_num)

        return self.state, reward, done

    def _gen_narrative_state(self, student_action):
        """
            helper function: create a narrative state from current student state for generating narrative action
        """
        narrative_state = deepcopy(self.state)
        narrative_state[26] = 0  # bryce trigger
        narrative_state[27] = 0  # teresa trigger
        narrative_state[28] = 0  # quiz trigger
        narrative_state[29] = 0  # diagnosis trigger

        if student_action == 14:  # quiz trigger
            narrative_state[28] = 1
        elif student_action == 15:  # bryce trigger
            narrative_state[26] = 1
        elif student_action == 17:  # teresa trigger
            narrative_state[27] = 1
        elif student_action == 18:  # diagnosis trigger
            narrative_state[29] = 1

        return narrative_state

    def _set_student_state(self, aes_action, aes_state_num):
        """
            helper function: create student state from narrative actions converted to aes actions.
            note, aes action ranges from [1-3]. but narrative action ranges from [0-9]
        """
        # resetting last AES
        self.state[19] = 0
        self.state[20] = 0
        self.state[21] = 0
        self.state[22] = 0

        self.state[aes_state_num] = aes_action

        return self.state

    def _gen_narrative_action(self, narrative_state, random=True) -> int:
        """
            this function generates a narrative action along with aes action and aes state position.
            when random is given, it generates uniform random narrative planner action
        """
        narrative_action = -1
        aes_state_num = -1
        aes_action = -1
        if self.narrative_planner is None or random:
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

        return narrative_action, aes_state_num, aes_action
