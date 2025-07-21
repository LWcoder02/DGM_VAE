import logging
import warnings
from os import PathLike
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import save_image, make_grid
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple
from core.utils.general import hyphenate_and_wrap_text

logger = logging.getLogger(__name__)


def generate_cvae_report(agent, artifacts_dir: PathLike = "cvae_report",
                         dataset_info: Optional[Dict[str, List]] = None, data_loader=None):
    """
    Generate images for the CVAE mode report

    """
    warnings.warn(
        "generate_cvae_report is deprecated. Use generate_hybrid_cvae_report instead. Code here is only for "
        "reference and will be removed in future versions.",
        DeprecationWarning,
        stacklevel=2
    )

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    generate_class_conditioned_samples_grid(
        agent=agent,
        dataset_info=dataset_info,
        samples_per_class=5,
        fig_size=(12, 8),
        artifacts_dir=artifacts_dir,
    )

    generate_class_conditioned_samples_with_same_latent_grid(
        agent=agent,
        dataset_info=dataset_info,
        n_latent_vectors=5,
        fig_size=(12, 8),
        artifacts_dir=artifacts_dir,
    )

    generate_reconstruction_comparison(
        agent=agent,
        test_dataloader=data_loader,
        dataset_info=dataset_info,
        artifacts_dir=artifacts_dir,
        n_samples=8,
        fig_size=(12, 4),
    )

    logger.info("Completed generating CVAE report.")


def generate_class_conditioned_samples_grid(
        agent, dataset_info, samples_per_class: int = 5, fig_size: Tuple[int, int] = (12, 8),
        artifacts_dir: PathLike = "outputs"
):
    """
    Generate a grid of conditional samples showing different classes.
    Each row represents one class, each column shows different samples of that class.
    """
    if not agent.is_conditional_training:
        logger.warning("Agent is not conditional. Skipping conditional generation.")
        return

    class_labels = dataset_info.get('label', {})
    n_classes = len(class_labels)
    dataset_name = dataset_info.get('python_class', 'Unknown Dataset')
    artifacts_dir = Path(artifacts_dir)

    # Generate samples for each class
    all_samples = []
    class_names = []

    for class_idx in range(n_classes):
        samples = agent.predict(num_samples=samples_per_class, labels=class_idx)
        all_samples.append(samples)
        class_names.append(class_labels.get(str(class_idx), f'Class {class_idx}'))

    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)

    grid = make_grid(all_samples, nrow=samples_per_class, normalize=True, padding=2, pad_value=1.0)

    # Save the grid image
    save_path = artifacts_dir / "cvae_conditional_samples"
    save_image(grid, save_path.with_suffix(".png"))
    save_image(grid, save_path.with_suffix(".pdf"))

    # Create annotated version with class labels
    fig, ax = plt.subplots(figsize=fig_size)

    # Convert tensor to numpy for matplotlib
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    if grid_np.shape[2] == 1:  # Grayscale
        grid_np = grid_np.squeeze(2)
        ax.imshow(grid_np, cmap='gray')
    else:
        ax.imshow(grid_np)

    # Add class labels on the left side
    img_height = grid_np.shape[0]
    row_height = img_height // n_classes

    # Calculate appropriate font size and wrap width based on number of classes
    label_fontsize, wrap_width = compute_wrap_width_and_label_font_size(n_classes)

    for i, class_name in enumerate(class_names):
        y_pos = (i + 0.5) * row_height

        # Join lines with newline characters
        display_name = wrap_class_name(class_name, wrap_width=wrap_width)

        ax.text(-40, y_pos, display_name, rotation=0, ha='right', va='center',
                fontsize=label_fontsize, fontweight='bold', color='black',
                linespacing=1.2)

    ax.set_title(f'CVAE Conditional Generation - Samples by Class on {dataset_name}', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Adjust subplot parameters to make room for multi-line labels
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)
    save_path = artifacts_dir / "cvae_conditional_samples_labeled"
    plt.savefig(save_path.with_suffix(".png"), bbox_inches='tight', format='png')
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches='tight', format='pdf')
    plt.close()

    logger.info(f"Conditional samples saved to {artifacts_dir}")


def generate_reconstruction_comparison(
        agent,
        test_dataloader,
        dataset_info: Optional[Dict[str, List]] = None,
        artifacts_dir: PathLike = "outputs",
        n_samples: int = 8,
        fig_size: Tuple[int, int] = (12, 8),
):
    """
    Generate reconstruction comparison using real test data images.
    Shows original images vs their reconstructions side by side.
    """
    agent._model.eval()
    device = agent._device
    artifacts_dir = Path(artifacts_dir)

    # Get a batch of test images
    test_batch = next(iter(test_dataloader))
    if agent.is_conditional_training:
        test_images, test_labels = test_batch
        test_images = test_images[:n_samples].to(device)
        test_labels = test_labels[:n_samples].to(device).squeeze(dim=1)
    else:
        test_images, _ = test_batch
        test_images = test_images[:n_samples].to(device)
        test_labels = None

    # Generate reconstructions
    with torch.no_grad():
        if agent.is_conditional_training:
            reconstructed, mu, logvar = agent._model(test_images, test_labels)
        else:
            reconstructed, mu, logvar = agent._model(test_images)

    # Create comparison visualization
    fig, axes = plt.subplots(2, n_samples, figsize=fig_size)

    # Get class labels for annotation (if available)
    dataset_name = dataset_info.get('python_class', 'Unknown Dataset')
    class_labels = dataset_info.get('label', {}) if agent.is_conditional_training else None
    n_classes = len(class_labels)
    label_fontsize, wrap_width = compute_wrap_width_and_label_font_size(n_classes)

    for i in range(n_samples):
        # Original image (top row)
        orig_img = test_images[i].cpu().permute(1, 2, 0).numpy()
        if orig_img.shape[2] == 1:  # Grayscale
            orig_img = orig_img.squeeze(2)
            axes[0, i].imshow(orig_img, cmap='gray')
        else:
            axes[0, i].imshow(orig_img)

        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].text(-0.1, 0.5, 'Original', transform=axes[0, i].transAxes,
                            rotation=90, ha='center', va='center', fontsize=12, fontweight='bold')

        # Add class label if available
        if class_labels and test_labels is not None:
            class_idx = test_labels[i].item()
            class_name = class_labels.get(str(class_idx), f'Class {class_idx}')
            wrapped_name = wrap_class_name(class_name, wrap_width=wrap_width)
            axes[0, i].set_title(wrapped_name, fontsize=label_fontsize, fontweight='bold')

        # Reconstructed image (bottom row)
        recon_img = reconstructed[i].cpu().permute(1, 2, 0).numpy()
        if recon_img.shape[2] == 1:  # Grayscale
            recon_img = recon_img.squeeze(2)
            axes[1, i].imshow(recon_img, cmap='gray')
        else:
            axes[1, i].imshow(recon_img)

        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].text(-0.1, 0.5, 'Reconstructed', transform=axes[1, i].transAxes,
                            rotation=90, ha='center', va='center', fontsize=12, fontweight='bold')

    # Calculate reconstruction loss for annotation
    mse_loss = torch.nn.functional.mse_loss(reconstructed, test_images, reduction='mean')

    model_type = "CVAE" if agent.is_conditional_training else "VAE"
    plt.suptitle(f'{model_type} Reconstruction Comparison on {dataset_name}\n'
                 f'Original vs Reconstructed Images (MSE Loss: {mse_loss:.4f})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, left=0.08)
    save_path = artifacts_dir / f"{model_type.lower()}_reconstruction_comparison"
    plt.savefig(save_path.with_suffix(".png"), bbox_inches='tight', format='png')
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches='tight', format='pdf')
    plt.close()


def generate_class_conditioned_samples_with_same_latent_grid(
        agent,
        dataset_info,
        n_latent_vectors: int = 5,
        fig_size: Tuple[int, int] = (12, 8),
        artifacts_dir: PathLike = "outputs"
):
    """
    Generate comparison showing same latent vectors with different conditional labels.
    """
    if not agent.is_conditional_training:
        logger.warning("Agent is not conditional. Skipping conditional reconstruction comparison.")
        return

    class_labels = dataset_info.get('label', {})
    n_classes = len(class_labels)
    dataset_name = dataset_info.get('python_class', 'Unknown Dataset')
    latent_dim = agent._model.latent_dim
    device = agent._device

    artifacts_dir = Path(artifacts_dir)

    # Generate fixed latent vectors
    with torch.no_grad():
        fixed_latent_vectors = torch.randn(n_latent_vectors, latent_dim).to(device)

        # Generate samples for each latent vector with each class condition
        all_samples = []

        for i, z in enumerate(fixed_latent_vectors):
            row_samples = []
            for class_idx in range(n_classes):
                # Use same latent vector but different class condition
                z_expanded = z.unsqueeze(0)  # Add batch dimension
                class_tensor = torch.tensor([class_idx]).to(device)
                sample = agent._model.decode(z_expanded, class_tensor)
                row_samples.append(sample)

            # Concatenate samples for this latent vector
            row_samples = torch.cat(row_samples, dim=0)
            all_samples.append(row_samples)

    # Create visualization
    fig, axes = plt.subplots(n_latent_vectors, n_classes, figsize=fig_size)

    # Handle single row case
    if n_latent_vectors == 1:
        axes = axes.reshape(1, -1)

    # Get class names for headers
    class_names = [class_labels.get(str(i), f'Class {i}') for i in range(n_classes)]

    # Calculate appropriate font size and wrap width based on number of classes
    label_fontsize, wrap_width = compute_wrap_width_and_label_font_size(n_classes)

    for row_idx, row_samples in enumerate(all_samples):
        for col_idx, sample in enumerate(row_samples):
            ax = axes[row_idx, col_idx]

            # Display sample
            sample_img = sample.cpu().permute(1, 2, 0).numpy()
            if sample_img.shape[2] == 1:  # Grayscale
                sample_img = sample_img.squeeze(2)
                ax.imshow(sample_img, cmap='gray')
            else:
                ax.imshow(sample_img)

            ax.axis('off')

            # Add class name as column header (only for first row)
            if row_idx == 0:
                # Wrap long class names into multiple lines
                wrapped_name = wrap_class_name(class_names[col_idx], wrap_width=wrap_width)

                ax.set_title(wrapped_name, fontsize=label_fontsize, fontweight='bold', pad=10)

            # Add latent vector label on the left (only for first column)
            if col_idx == 0:
                ax.text(-0.35, 0.5, f'Latent\nVector {row_idx + 1}',
                        transform=ax.transAxes, rotation=0, ha='center', va='center',
                        fontsize=9, fontweight='bold')

    plt.suptitle(f'CVAE Conditional Reconstruction Comparison on {dataset_name}\n'
                 'Same Latent Vectors with Different Class Conditions',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, left=0.1)

    save_path = artifacts_dir / "cvae_conditional_reconstruction_comparison"
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches='tight', format='png')
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches='tight', format='pdf')
    plt.close()

    # Also create a detailed version with annotations TODO
    # create_detailed_reconstruction_comparison(all_samples, class_names, artifacts_dir, n_latent_vectors)

    logger.info(f"Conditional reconstruction comparison saved to {artifacts_dir}")


def compute_wrap_width_and_label_font_size(n_classes: int) -> tuple[int, int]:
    """
    Compute proper label font size and wrap width based on number of classes.
    """
    if n_classes <= 5:
        label_fontsize = 10
        wrap_width = 12
    elif n_classes <= 10:
        label_fontsize = 9
        wrap_width = 10
    else:
        label_fontsize = 8
        wrap_width = 8
    return label_fontsize, wrap_width


def wrap_class_name(class_name: str, wrap_width: int = 10) -> str:
    """
    Wrap class name to fit within a specified width.
    :param class_name: The class name to wrap.
    :param wrap_width: The maximum width of each line.
    :return: Wrapped class name.
    """
    wrapped_lines = hyphenate_and_wrap_text(class_name, wrap_width=wrap_width)
    # If still too many lines, truncate and add ellipsis
    if len(wrapped_lines) > 3:
        wrapped_lines = wrapped_lines[:2] + [wrapped_lines[2][:wrap_width - 3] + "..."]

    return '\n'.join(wrapped_lines)


def generate_hybrid_cvae_report(
        agent,
        artifacts_dir: PathLike = "hybrid_cvae_report",
        dataset_info: Optional[Dict[str, List]] = None,
        multi_loader=None,
        test_datasets: Optional[Dict[str, torch.utils.data.Dataset]] = None,
):
    # Multi-dataset conditional generation grid
    generate_multi_dataset_samples_grid(
        agent=agent,
        multi_loader=multi_loader,
        artifacts_dir=artifacts_dir,
        samples_per_class=4
    )

    # # Cross-dataset reconstruction comparison
    # if test_datasets:
    #     generate_cross_dataset_reconstruction(
    #         agent=agent,
    #         test_datasets=test_datasets,
    #         artifacts_dir=artifacts_dir,
    #         n_samples=6
    #     )
    #
    #     # Individual dataset analysis
    #     for dataset_name, test_dataset in test_datasets.items():
    #         generate_dataset_specific_analysis(
    #             agent=agent,
    #             dataset_name=dataset_name,
    #             test_dataset=test_dataset,
    #             multi_loader=multi_loader,
    #             artifacts_dir=artifacts_dir
    #         )
    #
    # # Label type comparison
    # generate_label_type_comparison(
    #     agent=agent,
    #     multi_loader=multi_loader,
    #     artifacts_dir=artifacts_dir
    # )

    logger.info("Hybrid CVAE visual report generation completed.")


def generate_multi_dataset_samples_grid(agent, multi_loader, artifacts_dir: PathLike,
                                        samples_per_class: int = 4, fig_size: Tuple[int, int] = (12, 8)):
    """
    Generate individual sample grids for each dataset
    Layout: Each row represents one class, each column represents one sample of that class
    """
    if not agent.use_hybrid_conditioning:
        logger.warning("Agent does not use hybrid conditioning. Skipping multi-dataset generation.")
        return

    artifacts_dir = Path(artifacts_dir)

    # Generate samples for each dataset separately
    for dataset_id, dataset_name in enumerate(multi_loader.dataset_names):
        logger.info(f"Generating samples grid for dataset: {dataset_name}")

        dataset_info = multi_loader.datasets_info[dataset_name]
        label_type_info = multi_loader.label_type_info[dataset_name]

        dataset_samples = []
        class_names = []

        if label_type_info['type'] == 'single':
            # Generate samples for each class in single-label dataset
            n_classes = min(label_type_info['n_classes'], 8)  # Limit to 8 classes for visualization
            for class_id in range(n_classes):
                samples = agent.predict(
                    num_samples=samples_per_class,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    label_type='single',
                    labels=class_id
                )
                dataset_samples.append(samples)
                class_names.append(dataset_info['info']['label'].get(str(class_id), f'Class {class_id}'))
        else:
            # For multi-label, generate samples with different label combinations
            n_classes = label_type_info['n_classes']
            # Generate samples for first few individual labels
            for class_id in range(min(n_classes, 6)):  # Show more classes for multi-label
                labels = torch.zeros(samples_per_class, n_classes)
                labels[:, class_id] = 1.0
                samples = agent.predict(
                    num_samples=samples_per_class,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    label_type='multi',
                    labels=labels
                )
                dataset_samples.append(samples)
                class_names.append(f"Label {class_id}")

        if not dataset_samples:
            logger.warning(f"No samples generated for dataset {dataset_name}")
            continue

        n_classes_shown = len(class_names)

        # Create subplot grid: rows = classes, cols = samples + 1 (for label)
        n_cols = samples_per_class + 1  # +1 for label column
        n_rows = n_classes_shown

        # Calculate optimal figure size
        col_width = 1.0  # Width per sample column
        row_height = 1.0  # Height per class row
        label_col_width = 1.0  # Extra width for label column

        fig_width = label_col_width + (samples_per_class * col_width)
        fig_height = n_rows * row_height + 1.5  # +1.5 for title space

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(fig_width, fig_height),
                                 gridspec_kw={'width_ratios': [1.0] + [1] * samples_per_class})

        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # Plot each class and its samples
        for class_idx, (class_samples, class_name) in enumerate(zip(dataset_samples, class_names)):

            # First column: Class label
            axes[class_idx, 0].text(0.1, 0.5, wrap_class_name(class_name, wrap_width=15),
                                    ha='center', va='center', fontsize=11, fontweight='bold',
                                    transform=axes[class_idx, 0].transAxes)
            axes[class_idx, 0].axis('off')

            # Remaining columns: Individual samples
            for sample_idx in range(samples_per_class):
                sample = class_samples[sample_idx]

                # Convert tensor to numpy for display
                if sample.dim() == 3:  # [C, H, W]
                    sample_np = sample.permute(1, 2, 0).cpu().numpy()
                else:  # Handle other formats
                    sample_np = sample.cpu().numpy()

                # Display sample
                if sample_np.shape[-1] == 1 or len(sample_np.shape) == 2:  # Grayscale
                    if len(sample_np.shape) == 3:
                        sample_np = sample_np.squeeze(-1)
                    axes[class_idx, sample_idx + 1].imshow(sample_np, cmap='gray')
                else:  # RGB
                    axes[class_idx, sample_idx + 1].imshow(sample_np)

                axes[class_idx, sample_idx + 1].axis('off')

                # Add sample number as title for first row only
                if class_idx == 0:
                    axes[class_idx, sample_idx + 1].set_title(f'Sample {sample_idx + 1}',
                                                              fontsize=10, fontweight='bold')

        # Set main title
        label_type_str = label_type_info['type'].title()
        n_total_classes = label_type_info['n_classes']

        plt.suptitle(f'Dataset: {dataset_name}\n'
                     f'Type: {label_type_str}-Label | Classes: {n_total_classes} | '
                     f'Showing: {n_classes_shown} classes',
                     fontsize=14, fontweight='bold')

        # Adjust spacing
        plt.tight_layout()
        plt.subplots_adjust(
            top=0.90,  # Space for main title
            bottom=0.02,  # Minimal bottom margin
            left=0.05,  # Minimal left margin
            right=0.98,  # Minimal right margin
            hspace=0.1,  # Minimal vertical spacing between rows
            wspace=0.05  # Minimal horizontal spacing between columns
        )
        # ============ END NEW IMPLEMENTATION ============

        # Save individual dataset figure
        save_path = artifacts_dir / f"hybrid_cvae_samples_{dataset_name}"
        plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix(".pdf"), bbox_inches='tight')
        plt.close()

        logger.info(f"Sample grid for {dataset_name} saved to {save_path}")

    # Create a summary overview figure showing one example from each dataset
    _generate_dataset_overview(agent, multi_loader, artifacts_dir, samples_per_class)


def _generate_dataset_overview(agent, multi_loader, artifacts_dir: PathLike, samples_per_class: int = 4):
    """
    Generate a summary overview showing representative samples from each dataset
    """

    fig, axes = plt.subplots(1, len(multi_loader.dataset_names), figsize=(15, 3))

    if len(multi_loader.dataset_names) == 1:
        axes = [axes]

    for dataset_idx, dataset_name in enumerate(multi_loader.dataset_names):
        label_type_info = multi_loader.label_type_info[dataset_name]

        # Generate a few samples from the first class/label
        if label_type_info['type'] == 'single':
            samples = agent.predict(
                num_samples=samples_per_class,
                dataset_id=dataset_idx,
                dataset_name=dataset_name,
                label_type='single',
                labels=0
            )
        else:
            n_classes = label_type_info['n_classes']
            labels = torch.zeros(samples_per_class, n_classes)
            labels[:, 0] = 1.0
            samples = agent.predict(
                num_samples=samples_per_class,
                dataset_id=dataset_idx,
                dataset_name=dataset_name,
                label_type='multi',
                labels=labels
            )

        # Create grid
        grid = make_grid(samples, nrow=samples_per_class, normalize=True, padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()

        if grid_np.shape[2] == 1:
            grid_np = grid_np.squeeze(2)
            axes[dataset_idx].imshow(grid_np, cmap='gray')
        else:
            axes[dataset_idx].imshow(grid_np)

        axes[dataset_idx].axis('off')
        axes[dataset_idx].set_title(f'{dataset_name}\n({label_type_info["type"]}-label)',
                                    fontsize=12, fontweight='bold')

    plt.suptitle('Hybrid CVAE: Dataset Overview', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=1.5)  # Reduced padding
    plt.subplots_adjust(
        top=0.85,  # Space for title
        bottom=0.05,  # Minimal bottom margin
        left=0.05,  # Minimal left margin
        right=0.95,  # Minimal right margin
        hspace=0.3,  # Reduced vertical spacing between rows
        wspace=0.2  # Reduced horizontal spacing between columns
    )

    save_path = artifacts_dir / "hybrid_cvae_dataset_overview"
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches='tight')
    plt.close()

    logger.info(f"Dataset overview saved to {save_path}")


def generate_cross_dataset_reconstruction(agent, test_datasets: Dict, artifacts_dir: PathLike,
                                          n_samples: int = 6, fig_size: Tuple[int, int] = (15, 8)):
    """
    Generate cross-dataset reconstruction comparison showing original vs reconstructed
    images from multiple datasets in a unified view
    """
    artifacts_dir = Path(artifacts_dir)
    agent._model.eval()
    device = agent._device

    # Collect samples from each dataset
    dataset_comparisons = []
    dataset_names = []

    for dataset_name, test_dataset in test_datasets.items():
        # Create a temporary dataloader for this dataset
        from torch.utils.data import DataLoader
        from core.data.hybrid_dataset import collate_conditioned_samples

        temp_loader = DataLoader(test_dataset, batch_size=n_samples, shuffle=True,
                                 collate_fn=collate_conditioned_samples)

        # Get one batch
        batch_data = next(iter(temp_loader))
        images = batch_data['images'][:n_samples].to(device)

        # ============ FIXED: Create properly sliced batch data for condition kwargs ============
        # Slice all batch components to match the number of images
        limited_batch_data = {
            'images': batch_data['images'][:n_samples],
            'dataset_ids': batch_data['dataset_ids'][:n_samples],
            'dataset_names': batch_data['dataset_names'][:n_samples],
            'label_types': batch_data['label_types'][:n_samples],
            'single_mask': batch_data['single_mask'][:n_samples] if batch_data.get('single_mask') is not None else None,
            'multi_mask': batch_data['multi_mask'][:n_samples] if batch_data.get('multi_mask') is not None else None,
        }

        # Handle labels based on masks
        if limited_batch_data['single_mask'] is not None and limited_batch_data['single_mask'].any():
            single_count = limited_batch_data['single_mask'].sum().item()
            limited_batch_data['single_labels'] = batch_data['single_labels'][:single_count] if batch_data.get(
                'single_labels') is not None else None
        else:
            limited_batch_data['single_labels'] = None

        if limited_batch_data['multi_mask'] is not None and limited_batch_data['multi_mask'].any():
            multi_count = limited_batch_data['multi_mask'].sum().item()
            limited_batch_data['multi_labels'] = batch_data['multi_labels'][:multi_count] if batch_data.get(
                'multi_labels') is not None else None
        else:
            limited_batch_data['multi_labels'] = None

        # Generate reconstructions
        with torch.no_grad():
            condition_kwargs = agent._prepare_condition_kwargs(limited_batch_data)
            reconstructed, _, _ = agent._model(images, **condition_kwargs)
        # ============ END FIX ============

        # Create comparison (original + reconstructed)
        comparison = torch.cat([images, reconstructed], dim=0)
        dataset_comparisons.append(comparison.cpu())
        dataset_names.append(dataset_name)

    if not dataset_comparisons:
        logger.warning("No datasets available for cross-dataset reconstruction")
        return

    # Create visualization
    n_datasets = len(dataset_comparisons)
    fig, axes = plt.subplots(2 * n_datasets, n_samples, figsize=fig_size)

    if n_datasets == 1:
        axes = axes.reshape(2, -1)

    for dataset_idx, (comparison, dataset_name) in enumerate(zip(dataset_comparisons, dataset_names)):
        row_start = dataset_idx * 2

        # Original images
        for sample_idx in range(n_samples):
            img = comparison[sample_idx].permute(1, 2, 0).numpy()
            if img.shape[2] == 1:
                img = img.squeeze(2)
                axes[row_start, sample_idx].imshow(img, cmap='gray')
            else:
                axes[row_start, sample_idx].imshow(img)
            axes[row_start, sample_idx].axis('off')

            if sample_idx == 0:
                axes[row_start, sample_idx].text(-0.1, 0.5, f'{dataset_name}\nOriginal',
                                                 transform=axes[row_start, sample_idx].transAxes,
                                                 rotation=0, ha='right', va='center',
                                                 fontsize=10, fontweight='bold')

        # Reconstructed images
        for sample_idx in range(n_samples):
            img = comparison[sample_idx + n_samples].permute(1, 2, 0).numpy()
            if img.shape[2] == 1:
                img = img.squeeze(2)
                axes[row_start + 1, sample_idx].imshow(img, cmap='gray')
            else:
                axes[row_start + 1, sample_idx].imshow(img)
            axes[row_start + 1, sample_idx].axis('off')

            if sample_idx == 0:
                axes[row_start + 1, sample_idx].text(-0.1, 0.5, 'Reconstructed',
                                                     transform=axes[row_start + 1, sample_idx].transAxes,
                                                     rotation=0, ha='right', va='center',
                                                     fontsize=10, fontweight='bold')

    plt.suptitle('Hybrid CVAE: Cross-Dataset Reconstruction Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, top=0.9)

    save_path = artifacts_dir / "hybrid_cvae_cross_dataset_reconstruction"
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches='tight')
    plt.close()

    logger.info(f"Cross-dataset reconstruction comparison saved to {save_path}")


def generate_dataset_specific_analysis(agent, dataset_name: str, test_dataset, multi_loader,
                                       artifacts_dir: PathLike, n_samples: int = 8):
    """
    Generate detailed analysis for a specific dataset including:
    - Sample generation for all classes
    - Reconstruction quality analysis
    - Label-specific performance
    """
    artifacts_dir = Path(artifacts_dir)
    agent._model.eval()
    device = agent._device

    dataset_id = multi_loader.dataset_names.index(dataset_name)
    label_type_info = multi_loader.label_type_info[dataset_name]
    dataset_info = multi_loader.datasets_info[dataset_name]['info']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Dataset Analysis: {dataset_name}', fontsize=16, fontweight='bold')

    # 1. Generated samples grid (top-left)
    generated_samples = []
    if label_type_info['type'] == 'single':
        n_classes_to_show = min(label_type_info['n_classes'], 6)
        for class_id in range(n_classes_to_show):
            samples = agent.predict(
                num_samples=4,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                label_type='single',
                labels=class_id
            )
            generated_samples.append(samples)
    else:
        # Multi-label: show different combinations
        n_classes = label_type_info['n_classes']
        for i in range(min(4, n_classes)):
            labels = torch.zeros(4, n_classes)
            labels[:, i] = 1.0
            samples = agent.predict(
                num_samples=4,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                label_type='multi',
                labels=labels
            )
            generated_samples.append(samples)

    if generated_samples:
        all_generated = torch.cat(generated_samples, dim=0)
        grid = make_grid(all_generated, nrow=4, normalize=True, padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()

        if grid_np.shape[2] == 1:
            grid_np = grid_np.squeeze(2)
            axes[0, 0].imshow(grid_np, cmap='gray')
        else:
            axes[0, 0].imshow(grid_np)
        axes[0, 0].set_title('Generated Samples by Class')
        axes[0, 0].axis('off')

    # 2. Reconstruction comparison (top-right)
    from torch.utils.data import DataLoader
    from core.data.hybrid_dataset import collate_conditioned_samples

    temp_loader = DataLoader(test_dataset, batch_size=4, shuffle=True,
                             collate_fn=collate_conditioned_samples)
    batch_data = next(iter(temp_loader))

    images = batch_data['images'][:4].to(device)
    with torch.no_grad():
        # ============ FIXED: Create condition kwargs for the specific batch size ============
        # Need to slice the batch_data to match the number of images we're using
        limited_batch_data = {
            'images': batch_data['images'][:4],
            'dataset_ids': batch_data['dataset_ids'][:4],
            'dataset_names': batch_data['dataset_names'][:4],
            'label_types': batch_data['label_types'][:4],
            'single_mask': batch_data['single_mask'][:4] if batch_data.get('single_mask') is not None else None,
            'multi_mask': batch_data['multi_mask'][:4] if batch_data.get('multi_mask') is not None else None,
        }

        # Handle labels based on masks
        if batch_data.get('single_labels') is not None and batch_data['single_mask'][:4].any():
            single_count = batch_data['single_mask'][:4].sum().item()
            limited_batch_data['single_labels'] = batch_data['single_labels'][:single_count]
        else:
            limited_batch_data['single_labels'] = None

        if batch_data.get('multi_labels') is not None and batch_data['multi_mask'][:4].any():
            multi_count = batch_data['multi_mask'][:4].sum().item()
            limited_batch_data['multi_labels'] = batch_data['multi_labels'][:multi_count]
        else:
            limited_batch_data['multi_labels'] = None

        condition_kwargs = agent._prepare_condition_kwargs(limited_batch_data)
        reconstructed, _, _ = agent._model(images, **condition_kwargs)
        # ============ END FIX ============

    comparison = torch.cat([images, reconstructed], dim=0)
    grid = make_grid(comparison, nrow=4, normalize=True, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    if grid_np.shape[2] == 1:
        grid_np = grid_np.squeeze(2)
        axes[0, 1].imshow(grid_np, cmap='gray')
    else:
        axes[0, 1].imshow(grid_np)
    axes[0, 1].set_title('Reconstruction Comparison\n(Top: Original, Bottom: Reconstructed)')
    axes[0, 1].axis('off')

    # 3. Label distribution (bottom-left)
    class_labels = list(dataset_info['label'].values())
    class_counts = [1] * len(class_labels)  # Placeholder - in real scenario, count actual samples

    axes[1, 0].bar(range(len(class_labels)), class_counts)
    axes[1, 0].set_title('Class Distribution')
    axes[1, 0].set_xlabel('Classes')
    axes[1, 0].set_ylabel('Count')

    # Wrap long labels
    wrapped_labels = [wrap_class_name(label, wrap_width=8) for label in class_labels]
    axes[1, 0].set_xticks(range(len(class_labels)))
    axes[1, 0].set_xticklabels(wrapped_labels, rotation=45, ha='right', fontsize=8)

    # 4. Dataset info (bottom-right)
    info_text = f"""Dataset: {dataset_name}
Task: {dataset_info.get('task', 'N/A')}
Label Type: {label_type_info['type']}
N Classes: {label_type_info['n_classes']}
N Channels: {multi_loader.datasets_info[dataset_name]['n_channels']}
Image Size: {multi_loader.image_size}x{multi_loader.image_size}"""

    axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    axes[1, 1].axis('off')

    plt.tight_layout()

    save_path = artifacts_dir / f"hybrid_cvae_dataset_{dataset_name}_analysis"
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches='tight')
    plt.close()

    logger.info(f"Dataset-specific analysis for {dataset_name} saved to {save_path}")


def generate_label_type_comparison(agent, multi_loader, artifacts_dir: PathLike,
                                   samples_per_type: int = 8, fig_size: Tuple[int, int] = (14, 8)):
    """
    Generate comparison between single-label and multi-label datasets
    showing the differences in conditional generation
    """
    artifacts_dir = Path(artifacts_dir)

    # Separate datasets by label type
    single_label_datasets = []
    multi_label_datasets = []

    for dataset_name in multi_loader.dataset_names:
        label_type = multi_loader.label_type_info[dataset_name]['type']
        dataset_id = multi_loader.dataset_names.index(dataset_name)

        if label_type == 'single':
            single_label_datasets.append((dataset_id, dataset_name))
        else:
            multi_label_datasets.append((dataset_id, dataset_name))

    if not single_label_datasets and not multi_label_datasets:
        logger.warning("No datasets available for label type comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle('Label Type Comparison: Single-Label vs Multi-Label', fontsize=16, fontweight='bold')

    # Single-label samples (top row)
    if single_label_datasets:
        single_samples = []
        single_names = []

        for dataset_id, dataset_name in single_label_datasets[:2]:  # Limit to 2 datasets
            n_classes = min(multi_loader.label_type_info[dataset_name]['n_classes'], 4)
            for class_id in range(n_classes):
                samples = agent.predict(
                    num_samples=2,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    label_type='single',
                    labels=class_id
                )
                single_samples.append(samples)
            single_names.append(dataset_name)

        if single_samples:
            single_grid = make_grid(torch.cat(single_samples), nrow=8, normalize=True, padding=2)
            single_grid_np = single_grid.permute(1, 2, 0).cpu().numpy()

            if single_grid_np.shape[2] == 1:
                single_grid_np = single_grid_np.squeeze(2)
                axes[0, 0].imshow(single_grid_np, cmap='gray')
            else:
                axes[0, 0].imshow(single_grid_np)
            axes[0, 0].set_title(f'Single-Label Datasets\n{", ".join(single_names)}')
            axes[0, 0].axis('off')
    else:
        axes[0, 0].text(0.5, 0.5, 'No Single-Label\nDatasets', ha='center', va='center',
                        transform=axes[0, 0].transAxes, fontsize=14)
        axes[0, 0].axis('off')

    # Multi-label samples (top right)
    if multi_label_datasets:
        multi_samples = []
        multi_names = []

        for dataset_id, dataset_name in multi_label_datasets[:2]:  # Limit to 2 datasets
            n_classes = multi_loader.label_type_info[dataset_name]['n_classes']
            # Generate samples with different label combinations
            for i in range(min(4, n_classes)):
                labels = torch.zeros(2, n_classes)
                labels[:, i] = 1.0
                samples = agent.predict(
                    num_samples=2,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    label_type='multi',
                    labels=labels
                )
                multi_samples.append(samples)
            multi_names.append(dataset_name)

        if multi_samples:
            multi_grid = make_grid(torch.cat(multi_samples), nrow=8, normalize=True, padding=2)
            multi_grid_np = multi_grid.permute(1, 2, 0).cpu().numpy()

            if multi_grid_np.shape[2] == 1:
                multi_grid_np = multi_grid_np.squeeze(2)
                axes[0, 1].imshow(multi_grid_np, cmap='gray')
            else:
                axes[0, 1].imshow(multi_grid_np)
            axes[0, 1].set_title(f'Multi-Label Datasets\n{", ".join(multi_names)}')
            axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Multi-Label\nDatasets', ha='center', va='center',
                        transform=axes[0, 1].transAxes, fontsize=14)
        axes[0, 1].axis('off')

    # Statistics comparison (bottom row)
    stats_text = "Label Type Statistics:\n\n"
    stats_text += f"Single-Label Datasets: {len(single_label_datasets)}\n"
    stats_text += f"Multi-Label Datasets: {len(multi_label_datasets)}\n\n"

    if single_label_datasets:
        avg_single_classes = np.mean([multi_loader.label_type_info[name]['n_classes']
                                      for _, name in single_label_datasets])
        stats_text += f"Avg Classes (Single): {avg_single_classes:.1f}\n"

    if multi_label_datasets:
        avg_multi_classes = np.mean([multi_loader.label_type_info[name]['n_classes']
                                     for _, name in multi_label_datasets])
        stats_text += f"Avg Classes (Multi): {avg_multi_classes:.1f}\n"

    axes[1, 0].text(0.1, 0.5, stats_text, transform=axes[1, 0].transAxes,
                    fontsize=12, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
    axes[1, 0].axis('off')

    # Model architecture info
    model_text = "Model Configuration:\n\n"
    model_text += f"Hybrid Conditioning: {agent.use_hybrid_conditioning}\n"
    model_text += f"Latent Dimension: {agent._model.latent_dim}\n"
    model_text += f"Condition Dimension: {agent._model.condition_dim}\n"
    model_text += f"Total Datasets: {multi_loader.num_datasets}\n"
    model_text += f"Max Channels: {multi_loader.max_channels}\n"

    axes[1, 1].text(0.1, 0.5, model_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3))
    axes[1, 1].axis('off')

    plt.tight_layout()

    save_path = artifacts_dir / "hybrid_cvae_label_type_comparison"
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches='tight')
    plt.close()

    logger.info(f"Label type comparison saved to {save_path}")
