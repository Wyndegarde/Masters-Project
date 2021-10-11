
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import PercentFormatter

import torch
import torch.utils.data
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def load_in_data(res, data_path, batch_size, ratio=1):
    transform = transforms.Compose([
        transforms.Resize((res, res)),  # Resize images to 28*28
        transforms.Grayscale(),  # Make sure image is grayscale
        transforms.ToTensor()])  # change each image array to a tensor and scales inputs to [0,1]

    mnist_train = datasets.MNIST(data_path, train=True, download=True,
                                 transform=transform)  # Download training set and apply transformations.
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)  # same for test set

    train_len = int(len(mnist_train) / ratio)
    dummy_len = len(mnist_train) - train_len
    train_dataset, _ = random_split(mnist_train, (train_len, dummy_len), generator=torch.Generator().manual_seed(42))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader


def output_formula(input_size, filter_size, padding, stride):
    formula = math.floor((((input_size - filter_size + 2 * padding) / stride) + 1))

    return formula


def all_output_sizes(res, conv_filter=3, conv_padding=1, conv_stride=1, mp_filter=3, mp_padding=0, mp_stride=2):
    conv1 = output_formula(res, conv_filter, conv_padding, conv_stride)  # Output size from applying conv1 to input
    mp1 = output_formula(conv1, mp_filter, mp_padding, mp_stride)  # Output size from applying max pooling 1 to conv1

    conv2 = output_formula(mp1, conv_filter, conv_padding,
                           conv_stride)  # Output size from applying conv2 to max pooling 1
    conv3 = output_formula(conv2, conv_filter, conv_padding, conv_stride)  # Output size from applying conv3 to conv 2
    mp2 = output_formula(conv3, mp_filter, mp_padding, mp_stride)  # Output size from applying max pooling 2 to conv3

    conv4 = output_formula(mp2, conv_filter, conv_padding,
                           conv_stride)  # Output size from applying conv 4 to max pooling 2
    conv5 = output_formula(conv4, conv_filter, conv_padding, conv_stride)  # Output size from applying conv5 to conv 4
    mp3 = output_formula(conv5, mp_filter, mp_padding, mp_stride)  # Output size from applying max pooling 3 to conv 5

    outputs_i_need = [mp1, conv2, mp2, conv4, mp3]

    return outputs_i_need


def plot_training_history(history, res, loss_upper=1.05, acc_lower=-0.05, acc_higher=105):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(history['avg_train_loss'], label='train loss', marker='o')
    ax1.plot(history['avg_test_loss'], label='test loss', marker='o')

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylim([-0.05, loss_upper])
    ax1.legend()
    ax1.set_ylabel('Loss', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=16)

    ax2.plot(history['train_accuracy'], label='train accuracy', marker='o')
    ax2.plot(history['test_accuracy'], label='test accuracy', marker='o')

    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylim([acc_lower, acc_higher])

    ax2.legend()

    ax2.set_ylabel('Accuracy', fontsize=16)
    ax2.yaxis.set_major_formatter(PercentFormatter(100))
    ax2.set_xlabel('Epoch', fontsize=16)
    fig.suptitle(f'Training history ({res}*{res})', fontsize=20)
    plt.show()


def store_best_results(history):
    # Want to take the last entry from each output(best results) and store them all in a Dataframe
    placeholder = [history['avg_train_loss'][-1],
                   history['train_accuracy'][-1],
                   history['avg_test_loss'][-1],
                   history['test_accuracy'][-1]]

    return placeholder


def put_results_in_df(output):
    df = pd.DataFrame()
    df['avg_train_loss'] = output['avg_train_loss']
    df['train_accuracy'] = output['train_accuracy']
    df['avg_test_loss'] = output['avg_test_loss']
    df['test_accuracy'] = output['test_accuracy']

    return df
