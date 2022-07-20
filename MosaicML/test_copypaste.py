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
data_path = os.path.join(main_path, "examples", "crevasse", "data")
masks_path = os.path.join(data_path, "masks")
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
}

img_h = 400
img_l = 600

input_dict = {
     "sample_names": [],
     "masks": [],
     "images": []
}

sample_names = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]

trns = transforms.Compose([transforms.Resize((img_h, img_l)), transforms.ToTensor()])

for sample_name in sample_names:
     sample_path = os.path.join(data_path, sample_name)
     masks_path = os.path.join(sample_path, "masks")
     num_instances = len([name for name in os.listdir(masks_path) if name[-3:] == "png"]) + 1

     data = datasets.ImageFolder(sample_path, transform=trns)
     data_loader = torch.utils.data.DataLoader(data, batch_size=num_instances, shuffle=False)

     for i, data in enumerate(data_loader):
          input_dict["sample_names"].append(sample_name)
          input_dict["masks"].append(data[0][1:])
          input_dict["images"].append(data[0][0])


output_dict = copypaste_batch(input_dict, configs)

mt.save_copy_paste_output_dict(output_dict, main_path)

print(">>>>>INFO: Done")

