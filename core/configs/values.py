# This script is to store
from enum import Enum


class TypeEnum(Enum):
    """
    Base class for enums with string representation and fast lookup.
    """

    def __init__(self, value):
        self._id = value

    def __eq__(self, other):
        if isinstance(other, TypeEnum):
            return self._id == other._id
        elif isinstance(other, str):
            return self._id == other

    def __hash__(self):
        return hash(self._id)

    def __str__(self):
        """
        Return string representation of the enum.
        @return: String representation.
        """
        return self.value


class TrainingState(TypeEnum):
    """
    Enum for training states.
    """
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"  # completed state indicates successful completion of the training process
    FAILED = "failed"


class DataSplitType(TypeEnum):
    """
    Enum for data split types.
    """
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "val"

class VAEModelType(TypeEnum):
    """
    Enum for VAE model types.
    """
    VAE = "vae"
    CVAE = "cvae"


class DatasetLabelType(TypeEnum):
    """
    Enum for dataset label types.
    - single: the dataset has a single label for each sample
    - multi: the dataset has multiple labels for each sample
    """
    SINGLE = "single"
    MULTI = "multi"

class DatasetLabelInfoNames(TypeEnum):
    """
    Key names in dataset label info.
    - type: the type of the label (e.g. single, multi)
    - n_classes: the number of classes in the label
    """
    TYPE = "type"
    N_CLASSES = "n_classes"

class CheckpointCompressionFormat(TypeEnum):
    """
    Enum for checkpoint compression formats.
    """
    GZ = "gz"
    ZIP = "zip"


class OSSConfigKeys(TypeEnum):
    """
    Enum for OSS configuration keys.
    """
    ACCESS_KEY = "ACCESS_KEY_ID"
    SECRET_KEY = "SECRET_ACCESS_KEY"
    ENDPOINT = "ENDPOINT_URL"
    BUCKET_NAME = "BUCKET_NAME"
    REGION_NAME = "REGION"


class OSSURLType(TypeEnum):
    """
    Enum for OSS URL types.
    """
    PUBLIC = "public"
    PRIVATE = "private"
    SIGNED = "signed_url"
    CKPT = "checkpoint_id"

class DataInitialReturnNames(TypeEnum):
    """
    Enum for data initial return types.
    """
    CONDITION_INFO = "conditioning_info"
    HYBRID_DATALOADER = "hybrid_dataloader"
    TEST_MIXED = "test_mixed"
    TEST_SAMPLER = "test_sampler"
    TRAIN_MIXED = "train_mixed"
    TRAIN_SAMPLER = "train_sampler"
    VAL_MIXED = "val_mixed"
    VAL_SAMPLER = "val_sampler"
    TEST_DATASETS = "test_datasets"


