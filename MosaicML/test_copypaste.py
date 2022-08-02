import os 
import sys
from pprint import pprint
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from pathlib import Path


import mosaic_tools as mt 
from composer.algorithms.copypaste import copypaste_batch

pprint(sys.path)

################################
# update these paths if needed
main_path = os.path.join(".", "tools_utils", "MosaicML", "copypaste", "files")
data_path = os.path.join(main_path, "examples", "r1z2_samples")
masks_path = os.path.join(data_path, "all_masks")
image_path = os.path.join(data_path, "images")
################################

configs = {
     "convert_to_binary_mask": True,
     "p": 1.0,
     "max_copied_instances": None,
     "area_threshold": 100,
     "padding_factor": 0.5,
     "jitter_scale": (0.01, 0.99),
     "jitter_ratio": (1.0, 1.0),
     "p_flip": 1.0,
     "bg_color": 0
}

img_h = 512
img_l = 512


num_instances = len([name for name in os.listdir(os.path.join(image_path, "all")) if name[-3:] == "png"])
images = torch.zeros(num_instances, 3, img_h, img_l)
trns = transforms.Compose([transforms.Resize((img_h, img_l), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])
data = datasets.ImageFolder(image_path, transform=trns)
data_loader = torch.utils.data.DataLoader(data, batch_size=num_instances, shuffle=False)
for i, data in enumerate(data_loader):
     for i, image in enumerate(data[0]):
          images[i] = image


num_instances = len([name for name in os.listdir(os.path.join(masks_path, "all")) if name[-3:] == "png"])
masks = torch.zeros(num_instances, img_h, img_l)
trns = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((img_h, img_l), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])
data = datasets.ImageFolder(masks_path, transform=trns)
data_loader = torch.utils.data.DataLoader(data, batch_size=num_instances, shuffle=False)
for i, data in enumerate(data_loader):
     for i, mask in enumerate(data[0]):
          masks[i] = mask


# input_dict = {
#      "images": images,
#      "masks": masks
# }

out_images, out_masks = copypaste_batch(images, masks, configs)

# mt.save_copy_paste_output_dict(output_dict, main_path)

print(">>>>>INFO: Done")

