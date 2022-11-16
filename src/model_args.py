import json
from enum import Enum
import dataclasses
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    # generic params
    seed: int = field(default=34, metadata={"help": "random seed for reproducibility of results"})
    run_name: str = field(default="test_", metadata={"help": "the name of run. files will be saved by this name."})
    train_steps: int = field(default=int(1e5), metadata={"help": "maximum training time steps for all models"})
    gail_train_steps: int = field(default=int(1e3), metadata={"help": "maximum training time steps for gail only"})
    dryrun: bool = field(default=True, metadata={"help": "if we are saving actual files or not"})
    update_frequency: int = field(default=int(1e3), metadata={"help": "update target network frequency"})
    log_frequency: int = field(default=int(1e3), metadata={"help": "frequency of logs"})
    eval_only: bool = field(default=False, metadata={"help": "if we are just using for evaluation"})

    # Crystal Island env parameters
    state_dim: float = field(default=27, metadata={"help": "length of student state for crystal island"})
    action_dim: float = field(default=19, metadata={"help": "number of student actions for crystal island"})
    is_random_planner: bool = field(default=True, metadata={"help": "if the narrative planner will behave randomly or "
                                                                    "not"})
    # processed data locations
    student_data_loc: str = field(default="../processed_data/student_trajectories.pkl",
                                  metadata={"help": "location of the pickle file in pd.DataFrame"})
    narrative_data_loc: str = field(default="../processed_data/narrative_trajectories.pkl",
                                  metadata={"help": "location of the pickle file of narrative data in pd.DataFrame"})
    # model param
    device: str = field(default="cpu", metadata={"help": "device (cpu|cuda:0)"})
    units: float = field(default=128, metadata={"help": "number of neurons per layer"})
    lr: float = field(default=0.0001, metadata={"help": "learning rate"})
    internal_epoch_pi: int = field(default=10, metadata={"help": "training epochs of policy model"})
    internal_epoch_d: int = field(default=2, metadata={"help": "training epochs of discriminator model"})
    discount_factor: float = field(default=0.95, metadata={"help": "discount factor for gail"})
    clip_eps: float = field(default=0.2, metadata={"help": "clipping epsilon in PPO loss"})
    max_episode_len: int = field(default=500, metadata={"help": "maximum episode length"})
    simulate_episodes: int = field(default=5000, metadata={"help": "total number of episodes to simulate"})
    batch_size: int = field(default=256, metadata={"help": "batch size for training"})
    bcq_threshold: float = field(default=0.2, metadata={"help": "bcq model constraint parameter (tau value)"})
    validator_auth_threshold: float = field(default=.95, metadata={"help": "authenticator threshold for considering a "
                                                                           "valid episode"})

    # experimental
    validator_hidden_dim: int = field(default=64, metadata={"help": "hidden dim of lstm model"})
    validator_attn_head: int = field(default=4, metadata={"help": "attention head for lstm model"})
    validator_n_layers: int = field(default=2, metadata={"help": "lstm layers for lstm model"})
    validator_dropout: float = field(default=.1, metadata={"help": "dropout rate for lstm model"})

    # bcq narrative planner params


    # narrative planner params
    np_state_dim: float = field(default=31, metadata={"help": "length of narrative planner state for crystal island"})
    np_action_dim: float = field(default=10, metadata={"help": "number of narrative planner action for crystal island"})

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support).
        """
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)
