import logging
import sys
from os import PathLike
from pathlib import Path

from core.data.hybrid_dataset import init_dataloader, collate_conditioned_samples, MultiDatasetLoader

sys.path.insert(0, "../")

from typing import Tuple, Optional

from core.core import Core
from core.vae_agent import VariationalAutoEncoder, init_and_load_model

import torch
from torchvision.utils import save_image

from core.configs.values import DataInitialReturnNames as DRNames
from core.utils.general import set_random_seed, root_path
from core.configs.arguments import get_arguments, print_and_save_arguments
from core.configs.logging_config import setup_ml_logging_and_mlflow

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

logger = logging.getLogger(__name__)


def generate_samples(agent, num_samples=16, save_path: PathLike = "generated_samples.png"):
    samples_images = agent.predict(num_samples=num_samples)
    if save_path:
        save_image(samples_images, save_path, nrow=4, normalize=True)

def generate_hybrid_samples_and_reconstruct(
        agent,
        core: Core,
        multi_loader: MultiDatasetLoader,
        test_datasets:dict,
        artifacts_dir,
        mixed_dataset = None,
        sampler = None
):
    """Generate samples for hybrid label datasets"""

    if agent.use_hybrid_conditioning:
        for dataset_id, dataset_name in enumerate(multi_loader.dataset_names):
            logger.info(f"Generating samples for {dataset_name}")


            samples = agent.generate_dataset_specific_samples(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                num_samples_per_class=4
            )

            samples_save_path = Path(artifacts_dir, f"generated_samples_{dataset_name}.png")
            save_image(samples, samples_save_path, nrow=4, normalize=True)
            logger.info(f"Generated samples for {dataset_name} saved to {samples_save_path}")

    else:
        # Fallback to general sample generation
        samples_save_path = Path(artifacts_dir, "generated_samples.png")
        generate_samples(agent, num_samples=16, save_path=samples_save_path)
        logger.info(f"Generated samples saved to {samples_save_path}")

    # Test reconstruction on the mixed dataset
    if mixed_dataset:
        test_results = core.test(
            data=test_datasets,
            batch_size=8,
            collate_fn=collate_conditioned_samples,
            sampler=sampler
        )

        logger.info(f"Test results on mixed dataset: {test_results['test_loss(recon_loss)']}")
        if 'comparison' in test_results:
            comparison_save_path = Path(artifacts_dir, "reconstructed_comparison_mixed.png")
            save_image(test_results['comparison'], comparison_save_path, nrow=8, normalize=True)
            logger.info(f"Reconstructed images for mixed dataset saved to {comparison_save_path}")

    # Test reconstruction on each dataset
    all_comparisons = []
    for dataset_name, test_ds in test_datasets.items():
        logger.info(f"Testing reconstruction on {dataset_name}")
        test_results = core.test(
            data=test_ds,
            batch_size=8,
            collate_fn=collate_conditioned_samples
        )
        logger.info(f"Test results for {dataset_name}: {test_results['test_loss(recon_loss)']}")

        if 'comparison' in test_results:
            comparison_save_path = Path(artifacts_dir, f"reconstructed_comparison_{dataset_name}.png")
            save_image(test_results['comparison'], comparison_save_path, nrow=8, normalize=True)
            logger.info(f"Reconstructed images for {dataset_name} saved to {comparison_save_path}")
            all_comparisons.append(test_results['comparison'])

    # Create a combined comparison
    if len(all_comparisons) > 1:
        combined_comparison = torch.cat(all_comparisons[:3], dim=0)
        combined_save_path = Path(artifacts_dir, "reconstructed_comparison_all.png")
        save_image(combined_comparison, combined_save_path, nrow=8, normalize=True)
        logger.info(f"Combined reconstructed images saved to {combined_save_path}")


def setup_experiments():
    """
    Set up the experiment environment, including logging and MLflow.
    """
    args = get_arguments()
    set_random_seed(args.seed)
    root = root_path()

    log_dir, mlflow_logger = setup_ml_logging_and_mlflow(
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        disable_mlflow=True,
    )

    logger.info(f"Experiment setup complete with log directory: {log_dir}")

    return args, log_dir, mlflow_logger, logger, root


def run_evaluation():
    args, log_dir, mlflow_logger, logger, root = setup_experiments()
    timestamp = Path(log_dir).name

    logger = logging.getLogger(__name__)

    device = torch.device(args.device)

    if mlflow_logger:
        mlflow_logger.log_hyperparams(args)
        artifacts_dir = mlflow_logger.artifacts_dir
    else:
        if args.output:
            artifacts_dir = Path(args.output, 'artifacts')
        else:
            artifacts_dir = Path(root, "outputs", timestamp, 'artifacts')
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        print_and_save_arguments(args, save_dir=artifacts_dir)

    dataloader_kits = init_dataloader(args)
    conditioning_info = dataloader_kits[DRNames.CONDITION_INFO.value]
    hybrid_dataloader = dataloader_kits[DRNames.HYBRID_DATALOADER.value]
    test_datasets = dataloader_kits[DRNames.TEST_DATASETS.value]
    test_mixed = dataloader_kits[DRNames.TEST_MIXED.value]
    test_sampler = dataloader_kits[DRNames.TEST_SAMPLER.value]

    # Initialize model
    img_shape = (conditioning_info['unified_channels'], args.image_size, args.image_size)
    latent_dim = args.latent_dim
    logger.info(f"Model initialized with image shape {img_shape} and latent dimension {latent_dim}")

    agent = init_and_load_model(
        img_shape=img_shape,
        latent_dim=latent_dim,
        checkpoint_path=args.checkpoint_path,
        device=device,
        args=args,
        conditioning_info=conditioning_info
    )

    core = Core(agent=agent, optimizer=None, loss_function=None, num_workers=args.num_workers)

    generate_hybrid_samples_and_reconstruct(
        agent=agent,
        core=core,
        multi_loader=hybrid_dataloader,
        test_datasets=test_datasets,
        # mixed_dataset=test_mixed,
        artifacts_dir=artifacts_dir
    )

    if args.create_visual_report:
        core.generate_visual_report(
            artifacts_dir=artifacts_dir,
            dataset_info=conditioning_info,
            data=test_mixed if test_mixed else None,
            multi_loader=hybrid_dataloader,
            test_datasets=test_datasets,
        )
        # logger.info("Creating visual report for the model output...")
        # core.generate_visual_report(artifacts_dir=artifacts_dir, dataset_info=dataset_info, data=test_ds)


if __name__ == '__main__':
    run_evaluation()
