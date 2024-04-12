import os
import re
import numpy as np
from PIL import Image
from scipy import ndimage
from matplotlib import pyplot as plt

from config import (DATASET_NAME,
                    DATASET_TYPE,
                    SUMMARY_VIEW)


def plot_summary(checkpoint_dir, results_dir, epoch):
    # Show the nominated prediction image, target image, loss graph of that epoch, and metrics graph of that epoch

    img_index = SUMMARY_VIEW[DATASET_NAME][DATASET_TYPE]

    # Get prediction image

    results_contents = os.listdir(results_dir)

    prediction = None

    for result in results_contents:
        match = re.match(rf'e{epoch}_img{img_index}\.png', result)

        if match is not None:
            prediction = np.array(Image.open(f'{results_dir}/{result}'))
            break

    if prediction is None:
        print(f'No prediction image found in {results_dir}')
        return

    # Get target image

    target_paths = {
        'blender': f'./../data/nerf_synthetic/lego/test/r_{img_index}.png',
        'llff': f'./../data/nerf_llff_data/{DATASET_TYPE}/images_8/image{img_index:03d}.png'
    }

    target = np.array(Image.open(target_paths[DATASET_NAME]))

    # Blender target images are 800x800x4, need to convert to 400x400x3
    if DATASET_NAME == 'blender':
        target = ndimage.zoom(target[..., :3], (0.5, 0.5, 1))

    # Get epoch loss image

    loss_graph = np.array(Image.open(os.path.join(checkpoint_dir, f'loss_e{epoch}.png')))

    # Get epoch metrics image

    metrics_graph = np.array(Image.open(os.path.join(results_dir, f'e{epoch}_metrics_mean.png')))

    # Plot them

    fig, axs = plt.subplots(2, 2)

    fig.suptitle(f'Epoch {epoch} Summary')

    axs[0, 0].imshow(prediction)
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Reconstructed')

    axs[0, 1].imshow(target)
    axs[0, 1].axis('off')
    axs[0, 1].set_title('Ground Truth')

    axs[1, 0].imshow(loss_graph)
    axs[1, 0].axis('off')

    axs[1, 1].imshow(metrics_graph)
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
