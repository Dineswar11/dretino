import os

import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm import tqdm


def show_grid(image, title=None):
    """Create a grid of images
    Parameters
    ----------
    image : (np.array) Image
    title : (str, optional) Title.
    Defaults to None.
    """
    image = image.permute(1, 2, 0)

    image = np.clip(image, 0, 1)

    plt.figure(figsize=[15, 15])
    plt.imshow(image)
    if title is not None:
        plt.title(title)


def show_images(dataloader):
    """Show the image grid

    Parameters
    ----------
    dataloader : (DataLoader) DataLoader to get the images from 
    """
    class_labels = [0, 1, 2, 3, 4]
    data_iter = iter(dataloader)
    images, labels = data_iter.next()

    out = make_grid(images, nrow=4)

    show_grid(out, title=[class_labels[torch.argmax(x, dim=-1)] for x in labels])
    plt.axis('off')
    plt.show()


def cal_mean(loader):
    """Calculate Mean and Standard Deviation of the dataloader

    Parameters
    ----------
    loader : (DataLoader) DataLoader

    Returns
    -------
    mean : (long Tensor) Mean of the dataset
    std : (long Tensor) Std of the dataset
    """
    mean, std, total_img_count = 0, 0, 0

    for data, _ in tqdm(loader):
        img_count = data.size(0)
        data = data.float()
        data = data.view(img_count, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_img_count += img_count

    mean /= total_img_count
    std /= total_img_count

    return mean, std


def plot_metrics(file_name):
    """Plot the metrics for a given run

    Parameters
    ----------
    file_name : (str) Folder path of the run

    Returns
    -------
    plots : (plt) Train loss,Val loss Train accuracy,Val accuracy
    """
    for dirs, _, files in os.walk(file_name):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(dirs, file)

    metrics = pd.read_csv(path)

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss_epoch", "val_loss"]].plot(
        grid=True, legend=True, xlabel='Epoch', ylabel='Loss')
    df_metrics[["train_acc_epoch", "val_acc"]].plot(
        grid=True, legend=True, xlabel='Epoch', ylabel='Accuracy')
    plt.show()
