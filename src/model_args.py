import json
from enum import Enum
import dataclasses
from dataclasses import dataclass, field
import env.constants as envconst

@dataclass
class ModelArguments:
    # generic params
    seed: int = field(default=34, metadata={"help": "random seed for reproducibility of results"})
    run_name: str = field(default="test_", metadata={"help": "the name of run. files will be saved by this name."})
    train_steps: int = field(default=200, metadata={"help": "maximum training time steps for all models"})
    dryrun: bool = field(default=True, metadata={"help": "if we are saving actual files or not"})

    # Crystal Island env parameters
    state_dim: float = field(default=len(envconst.state_map), metadata={"help": "length of state for crystal island"})
    action_dim: float = field(default=len(envconst.action_map), metadata={"help": "number of actions for crystal island"})

    # processed data locations
    student_data_loc: str = field(default="../processed_data/student_trajectories.pkl",
                                  metadata={"help": "location of the pickle file in pd.DataFrame"})
    narrative_data_loc: str = field(default="../processed_data/narrative_trajectories.pkl",
                                  metadata={"help": "location of the pickle file of narrative data in pd.DataFrame"})
    # model param
    device: str = field(default="cpu", metadata={"help": "device (cpu|cuda:0)"})
    units: float = field(default=64, metadata={"help": "number of neurons per layer"})
    lr: float = field(default=0.0001, metadata={"help": "learning rate"})
    discount_factor: float = field(default=0.95, metadata={"help": "discount factor for gail"})
    clip_eps: float = field(default=0.3, metadata={"help": "clipping epsilon in PPO loss"})
    batch_size: int = field(default=256, metadata={"help": "batch size for training"})
    bcq_threshold: float = field(default=0.2, metadata={"help": "bcq model constraint parameter (tau value)"})

    epsilon: float = field(default=0.01, metadata={"help": "epsilon for exploring in gail"})
    cg_damping: float = field(default=0.1, metadata={"help": "cg_damping for gail"})
    max_kl: float = field(default=0.01, metadata={"help": "max_kl for gail"})
    lambda_: float = field(default=1e-3, metadata={"help": "lambda for gail"})

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
