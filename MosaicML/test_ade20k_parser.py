import os 
import sys
from pprint import pprint
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from pathlib import Path


import mosaic_tools as mt 


pprint(sys.path)

################################
# update these paths if needed
main_path = os.path.join(".", "tools_utils", "MosaicML", "copypaste", "files")
# data_path = os.path.join(main_path, "examples", "crevasse", "all_masks")
data_path = os.path.join(main_path, "examples", "tiny_ade20k_all_masks")
################################


img_h = 512
img_l = 735

input_dict = {
     "masks": [],
     "parsed_masks": []
}

sample_names = [name for name in os.listdir(os.path.join(data_path, "all_masks"))]

trns = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((img_h, img_l), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])


num_instances = len([name for name in os.listdir(os.path.join(data_path, "all_masks")) if name[-3:] == "png"])

data = datasets.ImageFolder(data_path, transform=trns)
data_loader = torch.utils.data.DataLoader(data, batch_size=num_instances, shuffle=False)

for i, data in enumerate(data_loader):
     for i, mask in enumerate(data[0]):
          input_dict["masks"].append(mask)


output_dict = mt.parse_segmentation_batch(input_dict)


print(">>>>>INFO: Done")

