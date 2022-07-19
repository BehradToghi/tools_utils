import os 
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

print(os.getcwd())

import tools_utils.MosaicML.mosaic_tools as mt
################################
# update these paths if needed
from forks.composer.composer.algorithms.copypaste import CopyPaste
from forks.composer.composer.algorithms.copypaste import copypaste_batch

main_path = os.path.join(".", "tools_utils", "MosaicML", "copypaste", "files")

data_path = os.path.join("examples", "crevasse", "data")

masks_path = os.path.join(main_path, data_path, "masks")
image_path = os.path.join(main_path, data_path, "images")
################################


img_h = 400
img_l = 600

input_dict = {
     "sample_names": [],
     "masks": [],
     "images": []
}

sample_names = [name for name in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, name))]

trns = transforms.Compose([transforms.Resize((img_h, img_l)), transforms.ToTensor()])

for sample_name in sample_names:
     sample_path = os.path.join(main_path, sample_name)
     masks_path = os.path.join(sample_path, "masks")
     num_instances = len([name for name in os.listdir(masks_path) if name[-3:] == "png"]) + 1

     data = datasets.ImageFolder(sample_path, transform=trns)
     data_loader = torch.utils.data.DataLoader(data, batch_size=num_instances, shuffle=False)

     for i, data in enumerate(data_loader):
          input_dict["sample_names"].append(sample_name)
          input_dict["masks"].append(data[0][1:])
          input_dict["images"].append(data[0][0])


output_dict = copypaste_batch(input_dict)

mt.save_copy_paste_output_dict(output_dict, main_path)

print(">>>>>INFO: Done")







