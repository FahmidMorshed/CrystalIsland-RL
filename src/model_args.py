import json
from enum import Enum
import dataclasses
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    # generic params
    seed: int = field(default=23, metadata={"help": "random seed for reproducibility of results"})
    run_type: str = field(default="train", metadata={"help": "type of run. For training GAIL use train. "
                                                             "For evaluation, use eval."})
    train_steps: int = field(default=1e7, metadata={"help": "maximum training time steps"})
    split_by: str = field(default="high", metadata={"help": "split dataset into high|low nlg"})

    # Crystal Island env parameters
    state_dim: float = field(default=26, metadata={"help": "length of student state for crystal island"})
    action_dim: float = field(default=19, metadata={"help": "number of student actions for crystal island"})
    is_random_planner: bool = field(default=True, metadata={"help": "if the narrative planner will behave randomly or "
                                                                    "not"})
    # processed data locations
    student_data_loc: str = field(default="../processed_data/student_trajectories.pkl",
                                  metadata={"help": "location of the pickle file in pd.DataFrame"})
    # model param
    device: str = field(default="cpu", metadata={"help": "device (cpu|cuda:0)"})
    units: float = field(default=64, metadata={"help": "number of neurons per layer"})
    lr_actor: float = field(default=0.001, metadata={"help": "actor model learning rate"})
    lr_critic: float = field(default=0.001, metadata={"help": "critic model learning rate"})
    lr_discriminator: float = field(default=0.001, metadata={"help": "discriminator model learning rate"})
    scheduler_gamma: float = field(default=0.99, metadata={"help": "exponential scheduler decrease rate"})
    internal_epoch_pi: int = field(default=20, metadata={"help": "training epochs of policy model"})
    internal_epoch_d: int = field(default=2, metadata={"help": "training epochs of discriminator model"})
    discount_factor: float = field(default=0.99, metadata={"help": "discount factor for gail"})
    clip_eps: float = field(default=0.2, metadata={"help": "clipping epsilon in PPO loss"})
    d_stop_threshold: float = field(default=0.1, metadata={"help": "maximum difference between expert score and "
                                                                    "novice score that is counted as a success"})
    d_stop_count: int = field(default=5, metadata={"help": "minimum consecutive number of times d_stop_threshold "
                                                           "is met"})
    max_episode_len: int = field(default=1500, metadata={"help": "maximum episode length"})
    update_steps: int = field(default=12000, metadata={"help": "frequency of model update"})

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