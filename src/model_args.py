import json
from enum import Enum
import dataclasses
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    # generic params
    seed: int = field(default=34, metadata={"help": "random seed for reproducibility of results"})
    run_name: str = field(default="test_", metadata={"help": "the name of run. files will be saved by this name."})
    gail_train_steps: int = field(default=1e3, metadata={"help": "maximum training time steps for gail"})
    dryrun: bool = field(default=True, metadata={"help": "if we are saving actual files or not"})
    load_validator: bool = field(default=False, metadata={"help": "load validator that is already saved"})
    load_gail: bool = field(default=False, metadata={"help": "load gail that is already saved"})
    load_sim: bool = field(default=False, metadata={"help": "load simulated data that is already saved"})
    debug: bool = field(default=False, metadata={"help": "running main in debug mode"})
    load_fqe: bool = field(default=False, metadata={"help": "load fqe estimator that is already saved"})
    load_bc: bool = field(default=False, metadata={"help": "load behavior cloning model that is already saved"})

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
    lr_actor: float = field(default=0.001, metadata={"help": "actor model learning rate"})
    lr_critic: float = field(default=0.001, metadata={"help": "critic model learning rate"})
    lr_discriminator: float = field(default=0.001, metadata={"help": "discriminator model learning rate"})
    scheduler_gamma: float = field(default=0.99, metadata={"help": "exponential scheduler decrease rate"})
    internal_epoch_pi: int = field(default=20, metadata={"help": "training epochs of policy model"})
    internal_epoch_d: int = field(default=2, metadata={"help": "training epochs of discriminator model"})
    discount_factor: float = field(default=0.99, metadata={"help": "discount factor for gail"})
    clip_eps: float = field(default=0.1, metadata={"help": "clipping epsilon in PPO loss"})
    max_episode_len: int = field(default=int(1e3), metadata={"help": "maximum episode length"})
    update_steps: int = field(default=int(1e3), metadata={"help": "frequency of model update"})
    simulate_episodes: int = field(default=int(1e3), metadata={"help": "total number of episodes to simulate"})


    # validator params
    lr_validator: float = field(default=0.001, metadata={"help": "validator model learning rate"})
    validator_train_steps: int = field(default=int(1e3), metadata={"help": "training epochs of validator model"})
    validator_batch: int = field(default=int(1e3), metadata={"help": "batch size for training auth validator"})
    validator_auth_threshold: float = field(default=.95, metadata={"help": "authenticator threshold for considering a "
                                                                           "valid episode"})

    # bcq narrative planner params
    bcq_batch: int = field(default=int(1e3), metadata={"help": "batch size for training bcq narrative planner"})
    lr_bcq: float = field(default=0.001, metadata={"help": "bcq model learning rate"})
    bcq_threshold: float = field(default=0.1, metadata={"help": "bcq model constraint parameter (tau value)"})
    bcq_train_steps: int = field(default=int(1e6), metadata={"help": "total number of training steps for the bcq"})
    bcq_update_frequency: int = field(default=int(1e3), metadata={"help": "update target network frequency"})
    behavior_cloning_train_steps: int = field(default=int(1e5), metadata={"help": "total number of training steps for "
                                                                               "behavior cloning"})
    bc_batch: int = field(default=int(1e3), metadata={"help": "batch size for training bc narrative planner"})

    # fqe params for doubly robust estimator
    fqe_train_steps: int = field(default=int(1e5), metadata={"help": "total number of training steps for the fqe"})
    lr_fqe: float = field(default=0.001, metadata={"help": "fqe model learning rate"})
    fqe_update_frequency: int = field(default=int(1e3), metadata={"help": "update target network frequency"})
    fqe_batch: int = field(default=int(1e3), metadata={"help": "batch size for training fqe narrative planner"})

    # narrative planner params
    np_state_dim: float = field(default=31, metadata={"help": "length of narrative planner state for crystal island"})
    np_action_dim: float = field(default=10, metadata={"help": "number of narrative planner action for crystal island"})
    np_discount: float = field(default=0.99, metadata={"help": "discounted factor or gamma for the narrative planner"})


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
