import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
from typing import List, Dict, Tuple, Optional
import logging
from torchvision import transforms
import torch.nn.functional as F

from core.data.dataset import load_medmnist_data
from core.configs.values import DataSplitType, DatasetLabelType, DatasetLabelInfoNames as DataLabelInfo, DataInitialReturnNames as DRNames
from core.utils.general import apply_smoke_test_settings

logger = logging.getLogger(__name__)


class DatasetConditionedSample:
    """Container for dataset-conditioned samples with multi-label support"""

    def __init__(self, image: torch.Tensor, dataset_id: int, labels: torch.Tensor,
                 dataset_name: str, label_type: str):
        self.image = image
        self.dataset_id = dataset_id
        self.labels = labels  # Can be single int or multi-label tensor
        self.dataset_name = dataset_name
        self.label_type = label_type  # 'single' or 'multi'


class MultiDatasetWrapper(Dataset):
    """Wrapper that adds dataset conditioning with multi-label support"""

    def __init__(self, dataset, dataset_id: int, dataset_name: str,
                 label_type: str, target_channels: int = None, target_size: int = None):
        self.dataset = dataset
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.label_type = label_type
        self.target_channels = target_channels
        self.target_size = target_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Standardize image format
        if self.target_channels is not None or self.target_size is not None:
            image = self._standardize_image(image)

        # Process labels based on type
        label = self._process_label(label)

        return DatasetConditionedSample(
            image=image,
            dataset_id=self.dataset_id,
            labels=label,
            dataset_name=self.dataset_name,
            label_type=self.label_type
        )

    def _process_label(self, label):
        """Process label based on single/multi type"""
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
        elif not isinstance(label, torch.Tensor):
            label = torch.tensor(label)

        if self.label_type == DatasetLabelType.SINGLE:
            return label.squeeze().long()
        else:  # multi-label
            return label.float()

    def _standardize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Standardize image to target channels and size"""
        # Handle channel dimension
        if self.target_channels is not None:
            current_channels = image.shape[0] if image.dim() == 3 else 1
            if current_channels != self.target_channels:
                if current_channels == 1 and self.target_channels == 3:
                    image = image.repeat(3, 1, 1)
                elif current_channels == 3 and self.target_channels == 1:
                    image = image.mean(dim=0, keepdim=True)
                else:
                    if current_channels < self.target_channels:
                        padding = torch.zeros(self.target_channels - current_channels, *image.shape[1:])
                        image = torch.cat([image, padding], dim=0)
                    else:
                        image = image[:self.target_channels]

        # Handle spatial dimensions
        if self.target_size is not None:
            current_size = image.shape[-1]
            if current_size != self.target_size:
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(self.target_size, self.target_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

        return image


class MultiDatasetLoader:
    """Enhanced loader with multi-label support"""

    def __init__(
            self,
            dataset_names: List[str],
            image_size: int = 28,
            as_rgb: bool = False,
            download: bool = True,
            dataset_weights: Optional[List[float]] = None
    ):
        self.dataset_names = dataset_names
        self.image_size = image_size
        self.as_rgb = as_rgb
        self.download = download
        self.dataset_weights = dataset_weights or [1.0] * len(dataset_names)

        self.datasets_info = {}
        self.raw_datasets = {}

        self.label_type_info = {}

        self._load_all_datasets()
        self._compute_unified_format()

    def _load_all_datasets(self):
        """Load all datasets and detect label types"""
        for dataset_name in self.dataset_names:
            logger.info(f"Loading dataset: {dataset_name}")

            train_ds, val_ds, test_ds, info = load_medmnist_data(
                dataset_name=dataset_name,
                image_size=self.image_size,
                download=self.download,
                as_rgb=self.as_rgb
            )

            self.raw_datasets[dataset_name] = {
                'train': train_ds, 'val': val_ds, 'test': test_ds
            }

            self.datasets_info[dataset_name] = {
                'info': info,
                'n_classes': len(info['label']),
                'n_channels': info['n_channels'],
                'task': info['task']
            }

            self.label_type_info[dataset_name] = {
                DataLabelInfo.TYPE.value: DatasetLabelType.MULTI.value if info.get('is_multi_label', False) else DatasetLabelType.SINGLE.value,
                DataLabelInfo.N_CLASSES.value: len(info['label'])
            }

            logger.info(f"  - Classes: {info['label']}")
            logger.info(f"  - Label type: {self.label_type_info[dataset_name][DataLabelInfo.TYPE.value]}")


    def _compute_unified_format(self):
        """Compute unified format with multi-label considerations"""
        self.max_channels = max(info['n_channels'] for info in self.datasets_info.values())
        self.num_datasets = len(self.dataset_names)

        TYPE_NAME = DataLabelInfo.TYPE.value
        N_CLASS_NAME = DataLabelInfo.N_CLASSES.value

        # max_single_classes is the maximum number of classes for single-label datasets, set it to 1 if no single-label datasets exist
        self.max_single_classes = max(
            info[N_CLASS_NAME] for name, info in self.label_type_info.items()
            if info[TYPE_NAME] == DatasetLabelType.SINGLE
        ) if any(info[TYPE_NAME] == DatasetLabelType.SINGLE for info in self.label_type_info.values()) else 1

        # max_multi_classes is the maximum number of classes for multi-label datasets, set it to 1 if no multi-label datasets exist
        self.max_multi_classes = max(
            info[N_CLASS_NAME] for name, info in self.label_type_info.items()
            if info[TYPE_NAME] == DatasetLabelType.MULTI
        ) if any(info[TYPE_NAME] == DatasetLabelType.MULTI for info in self.label_type_info.values()) else 1

        logger.info(f"Unified format - Channels: {self.max_channels}, "
                    f"Max single classes: {self.max_single_classes}, "
                    f"Max multi classes: {self.max_multi_classes}")

    def get_combined_dataset(self, split: str = 'train') -> Tuple[ConcatDataset, WeightedRandomSampler]:
        """Get combined dataset with label type awareness"""
        wrapped_datasets = []
        dataset_sizes = []

        for dataset_id, dataset_name in enumerate(self.dataset_names):
            dataset = self.raw_datasets[dataset_name][split]
            label_type = self.label_type_info[dataset_name][DataLabelInfo.TYPE.value]

            wrapped_dataset = MultiDatasetWrapper(
                dataset=dataset,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                label_type=label_type,
                target_channels=self.max_channels,
                target_size=self.image_size
            )

            wrapped_datasets.append(wrapped_dataset)
            dataset_sizes.append(len(wrapped_dataset))

        combined_dataset = ConcatDataset(wrapped_datasets)
        sampler = self._create_weighted_sampler(dataset_sizes)

        return combined_dataset, sampler

    def get_single_dataset(self, dataset_name: str, split: str = 'train') -> MultiDatasetWrapper:
        """Get single dataset with label type info"""
        if dataset_name not in self.dataset_names:
            raise ValueError(f"Dataset {dataset_name} not in loaded datasets")

        dataset_id = self.dataset_names.index(dataset_name)
        dataset = self.raw_datasets[dataset_name][split]
        label_type = self.label_type_info[dataset_name][DataLabelInfo.TYPE.value]

        return MultiDatasetWrapper(
            dataset=dataset,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            label_type=label_type,
            target_channels=self.max_channels,
            target_size=self.image_size
        )

    def _create_weighted_sampler(self, dataset_sizes: List[int]) -> WeightedRandomSampler:
        """Create weighted sampler"""
        sample_weights = []
        for dataset_id, (size, weight) in enumerate(zip(dataset_sizes, self.dataset_weights)):
            weight_per_sample = weight / size
            sample_weights.extend([weight_per_sample] * size)

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=sum(dataset_sizes),
            replacement=True
        )

    @property
    def conditioning_info(self) -> Dict:
        """Enhanced conditioning info with label types"""
        return {
            'num_datasets': self.num_datasets,
            'max_single_classes': self.max_single_classes,
            'max_multi_classes': self.max_multi_classes,
            'dataset_names': self.dataset_names,
            'unified_channels': self.max_channels,
            'datasets_info': self.datasets_info,
            'label_type_info': self.label_type_info
        }


def collate_conditioned_samples(batch: List[DatasetConditionedSample]) -> Dict[str, torch.Tensor]:
    images = torch.stack([sample.image for sample in batch])  # Stack images into a tensor [B, C, H, W]
    dataset_ids = torch.tensor([sample.dataset_id for sample in batch], dtype=torch.long) # [B]
    dataset_names = [sample.dataset_name for sample in batch]
    label_types = [sample.label_type for sample in batch]  # ['single', 'multi', ...] [B]

    # Separate single and multi-label samples
    single_mask = torch.tensor([lt == DatasetLabelType.SINGLE for lt in label_types])
    multi_mask = torch.tensor([lt == DatasetLabelType.MULTI for lt in label_types])

    # Process labels by type
    labels_dict = {}
    if single_mask.any():
        single_labels = [sample.labels for i, sample in enumerate(batch) if label_types[i] == DatasetLabelType.SINGLE.value]
        labels_dict['single_labels'] = torch.stack(single_labels) if single_labels else None

    if multi_mask.any():
        multi_labels = [sample.labels for i, sample in enumerate(batch) if label_types[i] == DatasetLabelType.MULTI.value]
        if multi_labels:
            # Pad multi-labels to same size
            max_classes = max(label.size(-1) for label in multi_labels)
            padded_labels = []
            for label in multi_labels:
                if label.size(-1) < max_classes:
                    padding = torch.zeros(max_classes - label.size(-1))
                    label = torch.cat([label, padding])
                padded_labels.append(label)
            labels_dict['multi_labels'] = torch.stack(padded_labels)
        else:
            labels_dict['multi_labels'] = None

    return {
        'images': images,
        'dataset_ids': dataset_ids,
        'dataset_names': dataset_names,
        'label_types': label_types,
        'single_mask': single_mask,
        'multi_mask': multi_mask,
        **labels_dict
    }


def init_dataloader(args):
    """
    Initialize the MedMNIST dataloader with the specified arguments.
    :param args:
    :return:
    - train_mixed: mixed all training datasets defined in args.dataset_names
    - test_datasets: dict of individual test datasets

    """
    # Load MedMNIST data
    hybrid_dataloader = MultiDatasetLoader(
        dataset_names=args.dataset_names,
        image_size=args.image_size,
        download=True,
        dataset_weights=args.dataset_weights,
    )
    conditioning_info = hybrid_dataloader.conditioning_info
    logger.info(f"Conditioning info: {conditioning_info}")
    # Get mixed dataset
    train_mixed, train_sampler = hybrid_dataloader.get_combined_dataset(DataSplitType.TRAIN.value)
    val_mixed, val_sampler = hybrid_dataloader.get_combined_dataset(DataSplitType.TRAIN.value)
    test_mixed, test_sampler = hybrid_dataloader.get_combined_dataset(DataSplitType.TEST.value)
    # Get individual test datasets
    test_datasets = {}
    for dataset_name in args.dataset_names:
        test_datasets[dataset_name] = hybrid_dataloader.get_single_dataset(dataset_name, DataSplitType.TEST.value)
    if args.smoke_test:
        logger.info("Running smoke test with minimal data")

        train_mixed, val_mixed, test_mixed = apply_smoke_test_settings(
            train_ds=train_mixed, val_ds=val_mixed, test_ds=test_mixed, args=args
        )
        train_sampler = val_sampler = None
    return {
        DRNames.CONDITION_INFO.value: conditioning_info,
        DRNames.HYBRID_DATALOADER.value: hybrid_dataloader,
        DRNames.TEST_DATASETS.value: test_datasets,
        DRNames.TRAIN_MIXED.value: train_mixed,
        DRNames.TRAIN_SAMPLER.value: train_sampler,
        DRNames.VAL_MIXED.value: val_mixed,
        DRNames.VAL_SAMPLER.value: val_sampler,
        DRNames.TEST_MIXED.value: test_mixed,
        DRNames.TEST_SAMPLER.value: test_sampler
    }