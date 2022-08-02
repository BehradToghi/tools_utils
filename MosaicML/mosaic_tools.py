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
    # plt.show()


def imshow_1ch_tensor(img):
    plt.figure()
    npimg = img.numpy()
    plt.imshow(npimg, cmap="gray")
    # plt.show()


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


def visualize_mask_parser(input_dict):
    for i in range(len(input_dict["masks"])):
        imshow_3ch_tensor(input_dict["masks"][i])
        parsed_masks = torch.unsqueeze(input_dict["parsed_masks"][i], dim=1)
        imshow_3ch_tensor(torchvision.utils.make_grid(parsed_masks, nrow=4))



def decompose_mask(mask, mask_color, background_color):
    parsed_mask = mask
    # imshow_1ch_tensor(torch.squeeze(mask))
    mask_npy = mask.numpy()
    unique_vals = np.unique(mask_npy) 


    parsed_mask = torch.zeros([len(unique_vals), mask.size(dim=1), mask.size(dim=2)])

    for i, val in enumerate(unique_vals):
        # print("val = ", val)
        temp_mask = torch.where(mask == val, mask_color, background_color)
        # imshow_1ch_tensor(torch.squeeze(temp_mask))

        parsed_mask[i] = temp_mask

    

    return parsed_mask




def parse_segmentation_batch(input_dict, mask_color=1, background_color=0):
    print("parsing")


    for i, mask in enumerate(input_dict["masks"]):
        input_dict["parsed_masks"].append(decompose_mask(mask, mask_color, background_color))


    # visualize_mask_parser(input_dict)

    return input_dict




#####################

def imshow_tensor(tensor):
    plt.figure()
    arr = np.transpose(tensor.cpu().numpy(), (1, 2, 0))
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    plt.imshow(arr)
    

def imshow_1d_tensor(tensor):
    plt.figure()
    arr = tensor.cpu().numpy()
    # arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    plt.imshow(arr+1, cmap="gray")
    


def visualize_copypaste_sample(src_image, src_mask, trg_image, trg_mask, out_image, out_mask, fig_name=None):
    if fig_name is None:
        fig_name = "copypaste_sample_test"

    dpi = 100
    fig, axarr = plt.subplots(2, 3, figsize=(10, 6), dpi=dpi)


    arr = src_image.cpu().numpy()
    ax = axarr[0, 0]
    ax.set_title("src_image")
    image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    ax.imshow(np.transpose(image, (1, 2, 0)))
    clean_axes(ax)

    arr = src_mask.cpu().numpy()
    ax = axarr[1, 0]
    ax.set_title("src_mask")
    image = arr + 1
    ax.imshow(image, cmap= "gray")
    clean_axes(ax)


    arr = trg_image.cpu().numpy()
    ax = axarr[0, 1]
    ax.set_title("trg_image")
    image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    ax.imshow(np.transpose(image, (1, 2, 0)))
    clean_axes(ax)

    arr = trg_mask.cpu().numpy()
    ax = axarr[1, 1]
    ax.set_title("trg_mask")
    image = arr + 1
    ax.imshow(image, cmap= "gray")
    clean_axes(ax)


    arr = out_image.cpu().numpy()
    ax = axarr[0, 2]
    ax.set_title("out_image")
    image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    ax.imshow(np.transpose(image, (1, 2, 0)))
    clean_axes(ax)

    arr = out_mask.cpu().numpy()
    ax = axarr[1, 2]
    ax.set_title("out_mask")
    image = arr + 1
    ax.imshow(image, cmap= "gray")
    clean_axes(ax)

    plt.suptitle("CopyPaste Augmentation Sample", fontweight="bold")
    fig_out_path = os.path.join(".", "debug_out", "samples")
    
    if not os.path.isdir(fig_out_path):
        os.makedirs(fig_out_path)
    print("sample image saved: ", fig_name)

    plt.savefig(os.path.join(fig_out_path, fig_name + ".png"), dpi=dpi)



def visualize_copypaste_batch(images, masks, out_images, out_masks, num, fig_name=None):
    if fig_name is None:
        fig_name = "copypaste_test"
    start_index = 50
    dpi = 200
    fig, axarr = plt.subplots(4, num, figsize=(14, 8), dpi=dpi)

    for col in range(min(num, len(images))):
        arr = images[start_index + col].cpu().numpy()
        image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        ax = axarr[0, col]
        ax.imshow(np.transpose(image, (1, 2, 0)))
        ax.set_title("images")
        clean_axes(ax)

    for col in range(min(num, len(masks))):
        arr = torch.unsqueeze(masks[start_index + col], dim=0).cpu().numpy()
        image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        ax = axarr[1, col]
        ax.imshow(np.transpose(image, (1, 2, 0)), cmap="gray")
        ax.set_title("masks")
        clean_axes(ax)

    for col in range(min(num, len(out_images))):
        arr = out_images[start_index + col].cpu().numpy()
        # print("out_images: ", (np.max(arr) - np.min(arr)))
        image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        ax = axarr[2, col]
        ax.imshow(np.transpose(image, (1, 2, 0)))
        ax.set_title("out_images")
        clean_axes(ax)

    for col in range(min(num, len(out_masks))):
        arr = torch.unsqueeze(out_masks[start_index + col], dim=0).cpu().numpy()
        # print("out_masks: ", (np.max(arr) - np.min(arr)))
        image = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        ax = axarr[3, col]
        ax.imshow(np.transpose(image, (1, 2, 0)), cmap="gray")
        ax.set_title("out_masks")
        clean_axes(ax)

    plt.suptitle("CopyPaste Augmentation Batch", fontweight="bold")
    fig_out_path = os.path.join(".", "debug_out")
    
    if not os.path.isdir(fig_out_path):
        os.makedirs(fig_out_path)

    plt.savefig(os.path.join(fig_out_path, fig_name + ".png"), dpi=dpi)


def clean_axes(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    ax.axis(('tight'))
    ax.set_aspect(aspect=1)


def save_tensor_to_png(tensor, path, name):
    arr = np.transpose(tensor.cpu().numpy(), (1, 2, 0))
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    plt.imsave(os.path.join(path, name), arr)
    print("Torch tensor saved to png: " + name)


def save_1d_tensor_to_png(tensor, path, name):
    arr = tensor.cpu().numpy()
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    plt.imsave(os.path.join(path, name), arr, cmap="gray")
    print("Torch tensor saved to png: " + name)


        # out_images[batch_idx] = trg_image
        # out_masks[batch_idx] = trg_mask
        
        # # fig_name = "copypaste_sample_test" + str(batch_idx)
        # # visualize_copypaste_sample(images[i], masks[i], images[j], masks[j], trg_image, trg_mask, fig_name=fig_name)