from src.model import nets


class ModularDQN:
    def __init__(self, args: dataclasses):
        self.args = args
        self.q_network = nets.QNetwork(self.args)