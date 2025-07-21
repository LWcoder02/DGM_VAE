import logging
import sys
from os import PathLike
from pathlib import Path
from typing import List, Optional

from torch.utils.data import DataLoader

from core.configs.arguments import get_arguments, print_and_save_arguments
from core.configs.logging_config import setup_ml_logging_and_mlflow
from core.configs.values import DataInitialReturnNames as DRNames
from core.data.hybrid_dataset import MultiDatasetLoader, collate_conditioned_samples, init_dataloader
from core.utils.general import set_random_seed, root_path, symlink_force, apply_smoke_test_settings

sys.path.insert(0, '../')

from core.core import Core
from core.loss_function import LossFunction
from core.optimizer import Optimizer
from core.vae_agent import init_and_load_model
from core.utils.saving import save_metrics, save_model
from core.visualization.plotting import plot_data

from core.loss_function import VAELoss

import torch

from torchvision.utils import save_image
import pandas as pd

logger = logging.getLogger(__name__)


def generate_samples(agent, num_samples=16, save_path: PathLike = "generated_samples.png"):
    samples_images = agent.predict(num_samples=num_samples)
    if save_path:
        save_image(samples_images, save_path, nrow=4, normalize=True)


def generate_random_samples_and_reconstruct_images(agent, core: Core, test_ds, artifacts_dir):
    # Generate and save samples
    samples_save_path = Path(artifacts_dir, "generated_samples.png")
    generate_samples(agent, num_samples=16, save_path=samples_save_path)
    logger.info(f"Generated samples saved to {samples_save_path}")

    # Reconstruct images from test set
    # Only reconstruct images from the first batch of the test dataset
    test_results = core.test(data=test_ds, batch_size=8)

    logger.info(f"Test results: {test_results['test_loss(recon_loss)']}")
    if 'comparison' in test_results:
        comparison_save_path = Path(artifacts_dir, "reconstructed_comparison.png")
        save_image(test_results['comparison'], comparison_save_path, nrow=8, normalize=True)
        logger.info(f"Reconstructed images saved to {comparison_save_path}")


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

    # Create combined comparison
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
        disable_mlflow=args.disable_mlflow,
    )

    logger.info(f"Experiment setup complete with log directory: {log_dir}")

    return args, log_dir, mlflow_logger, logger, root


def run_vae_experiment():
    args, log_dir, mlflow_logger, logger, root = setup_experiments()

    device = torch.device(args.device)

    timestamp = Path(log_dir).parts[-1]

    # Log parameters
    if mlflow_logger:
        mlflow_logger.log_hyperparams(args)
        artifacts_dir = mlflow_logger.artifacts_dir
    else:
        artifacts_dir = Path(root, "outputs", timestamp, 'artifacts')
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        print_and_save_arguments(args, save_dir=artifacts_dir)

    dataloader_kits = init_dataloader(args)
    conditioning_info = dataloader_kits[DRNames.CONDITION_INFO.value]
    train_mixed = dataloader_kits[DRNames.TRAIN_MIXED.value]
    train_sampler = dataloader_kits[DRNames.TRAIN_SAMPLER.value]
    val_mixed = dataloader_kits[DRNames.VAL_MIXED.value]
    val_sampler = dataloader_kits[DRNames.VAL_SAMPLER.value]
    hybrid_dataloader = dataloader_kits[DRNames.HYBRID_DATALOADER.value]
    test_datasets = dataloader_kits[DRNames.TEST_DATASETS.value]
    test_mixed = dataloader_kits[DRNames.TEST_MIXED.value]
    test_sampler = dataloader_kits[DRNames.TEST_SAMPLER.value]

    # Initialize model
    img_shape = (conditioning_info['unified_channels'], args.image_size, args.image_size)
    # n_classes = len(dataset_info['label'])
    # is_multi_label = dataset_info['is_multi_label']
    latent_dim = args.latent_dim
    loss_module = VAELoss(beta=args.beta)
    logger.info(f"Model initialized with image shape {img_shape} and latent dimension {latent_dim}")

    agent = init_and_load_model(img_shape=img_shape, latent_dim=latent_dim, checkpoint_path=args.checkpoint_path,
                                device=device, args=args, conditioning_info=conditioning_info)

    optimizer = Optimizer(optimizer=args.optimizer,
                          model_parameters=agent.get_parameters(),
                          config={'lr': args.learning_rate, "weight_decay": args.weight_decay})
    loss_function = LossFunction(loss_function=loss_module,
                                 device=device)
    core = Core(agent=agent, optimizer=optimizer, loss_function=loss_function, num_workers=args.num_workers)

    training_metrics, val_metrics = core.train(
        training_data=train_mixed,
        evaluation_data=val_mixed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        collate_fn=collate_conditioned_samples,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
    )
    logger.info("Training completed")

    # Save model parameters
    if args.save_model:
        save_model(agent=agent, root=root, timestamp=timestamp, mlflow_logger=mlflow_logger, args=args)

    # Save training metrics
    save_metrics(metrics=training_metrics, save_path=artifacts_dir, file_name="train")
    save_metrics(metrics=val_metrics, save_path=artifacts_dir, file_name="validation")

    # Plot training and validation losses
    plot_data(pd.DataFrame(list(zip(training_metrics["epoch"], training_metrics['loss'])),
                           columns=["Epoch", "Loss"]),
              save_path=artifacts_dir, file_name="losses_train", title="Training Losses", y_name="Loss")
    plot_data(pd.DataFrame(list(zip(val_metrics["epoch"], val_metrics['loss'])),
                           columns=["Epoch", "Loss"]), save_path=artifacts_dir, file_name="losses_validation",
              y_name="Loss")
    # Symlink the latest artifacts
    latest_artifacts_link = Path(root, "outputs", 'latest', 'artifacts')
    symlink_force(artifacts_dir, latest_artifacts_link)

    generate_hybrid_samples_and_reconstruct(
        agent=agent,
        core=core,
        multi_loader=hybrid_dataloader,
        test_datasets=test_datasets,
        # mixed_dataset=test_mixed,
        artifacts_dir=artifacts_dir
    )

if __name__ == '__main__':
    run_vae_experiment()

## Legacy code, kept for reference in order to grasp the framework structure and flow
# def run_experiment():
#     torch.manual_seed(0)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(device)
#
#     X, y = make_circles(n_samples=5000, noise=0.03, random_state=0)
#
#     network = NeuralNet(in_features=2, out_features=1)
#     loss_module = VAELoss()
#
#     agent = VariationalAutoEncoder(model=network, device=device)
#     optimizer = Optimizer(optimizer="SGD",
#                           model_parameters=agent.get_parameters(),
#                           config={'lr': 1e-1, "momentum": 0.9})
#     loss_function = LossFunction(loss_function=loss_module,
#                                  device=device)
#     core = Core(agent=agent, optimizer=optimizer, loss_function=loss_function)
#
#     X_train, y_train, X_test, y_test = core.split_data_train_test(X, y)
#     X_train, y_train, X_val, y_val = core.split_data_train_test(X_train, y_train)
#
#     train_data = core.build_dataset(X_train, y_train, device=device)
#     validation_data = core.build_dataset(X_val, y_val, device=device)
#     test_data = core.build_dataset(X_test, y_test, device=device)
#
#     training_metrics, val_metrics = core.train(training_data=train_data, evaluation_data=validation_data,
#                                                       batch_size=32)
#
#     plot_data(pd.DataFrame(list(zip(training_metrics["epoch"], training_metrics['loss'])), columns=["Epoch", "Loss"]),
#               save_path="experiments", file_name="losses_train", title="Losses", y_name="Loss")
#
#     plot_data(pd.DataFrame(list(zip(val_metrics["epoch"], val_metrics['loss'])), columns=["Epoch", "Loss"]),
#               save_path="experiments", file_name="losses_validation", title="Losses", y_name="Loss")
#
#     save_metrics(metrics=training_metrics, save_path="experiments", file_name="train")
#     save_metrics(metrics=val_metrics, save_path="experiments", file_name="validation")
