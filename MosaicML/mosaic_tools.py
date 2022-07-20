import os 
import glob
from random import sample, shuffle
import cv2
from PIL import Image
import numpy as np
from pip import main
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


def imshow_3ch_tensor(img):
    plt.figure()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def imshow_1ch_tensor(img):
    plt.figure()
    npimg = img.numpy()
    plt.imshow(npimg, cmap="gray")
    plt.show()


def save_tensor_to_png(tensor, path, name):

    arr = np.transpose(tensor.numpy(), (1, 2, 0))

    if not os.path.isdir(path):
        os.makedirs(path)
    plt.imsave(os.path.join(path, name), arr)
    print("Torch tensor saved to png: " + path + "/" + name)


def save_copy_paste_output_dict(output_dict, main_path):
    path = os.path.join(main_path, "out", "results")

    batch_size = len(output_dict["images"])
      
    for i in range(batch_size):
        save_tensor_to_png(output_dict["images"][i], os.path.join(path, str(i), "image"), str(i)+".png")


        x = output_dict["masks"]
        save_tensor_to_png(torchvision.utils.make_grid(x[i], padding=50, pad_value=0.85, nrow=3), os.path.join(path, str(i), "masks"), str(i)+ "_all.png")

        for j, mask in enumerate(output_dict["masks"][i]):
            save_tensor_to_png(mask, os.path.join(path, str(i), "masks"), str(i)+ "_" + str(j) + ".png")


def visualize_copypaste_batch(output_dict, input_dict, batch_size, i, j, fig_name=None):
    if fig_name is None:
        fig_name = "SRC-"+input_dict["sample_names"][i]+"-TRG-"+input_dict["sample_names"][j]

    dpi = 150

    fig, axarr = plt.subplots(2, batch_size, figsize=(18, 7), dpi=dpi)

    for col, image in enumerate(output_dict["images"]):

        ax = axarr[1, col]
        ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        clean_axes(ax)

    for col, image in enumerate(input_dict["images"]):
        ax = axarr[0, col]
        ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        clean_axes(ax)
        

    plt.suptitle("CopyPaste Augmentation Batch", fontweight="bold")

    fig_out_path = os.path.join(".", "forks", "composer", "composer", "algorithms", "copypaste", "files", "out", "no_jittering", fig_name)
    plt.savefig(fig_out_path + ".png", dpi=dpi)


def visualize_copypaste_instance(src_image, src_instance, trg_image_before, trg_image_after, input_dict, src_instance_id, i, j, fig_name):
    if fig_name is None:
        fig_name = "SRC-"+input_dict["sample_names"][i]+"-id-"+str(src_instance_id)+"-TRG-"+input_dict["sample_names"][j]
        
    dpi = 100
    img_1 = np.transpose(src_image.numpy(), (1, 2, 0))
    img_2 = np.transpose(src_instance.numpy(), (1, 2, 0))
    img_3 = np.transpose(trg_image_before.numpy(), (1, 2, 0))
    img_4 = np.transpose(trg_image_after.numpy(), (1, 2, 0))

    fig, axarr = plt.subplots(2, 2, figsize=(6, 6), dpi=dpi)
    
    ax = axarr[0, 0]
    ax.imshow(img_1)
    clean_axes(ax)
    ax.set_title("Source Image")
    ax = axarr[0, 1]
    ax.imshow(img_2)
    clean_axes(ax)
    ax.set_title("Source Instance")
    ax = axarr[1, 0]
    ax.imshow(img_3)
    clean_axes(ax)
    ax.set_title("Target Image")
    ax = axarr[1, 1]
    ax.imshow(img_4)
    clean_axes(ax)
    ax.set_title("Augmented Image")

    plt.suptitle("CopyPaste Augmentation Instance", fontweight="bold")

    fig_out_path = os.path.join(".", "forks", "composer", "composer", "algorithms", "copypaste", "files", "out", "no_jittering", fig_name)
    plt.savefig(fig_out_path + ".png", dpi=dpi)

def clean_axes(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    ax.axis(('tight'))
    ax.set_aspect(aspect=1)
