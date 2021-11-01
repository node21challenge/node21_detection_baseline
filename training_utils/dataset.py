import os
import numpy as np
import torch
from PIL import Image
import pandas as pd
import SimpleITK as sitk

import training_utils.transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class CXRNoduleDataset(object):
    def __init__(self, root, csv_file, transforms):
        self.root = root
        self.transforms = transforms
        self.data = pd.read_csv(csv_file)
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.imgs = [i for i in self.imgs if i in self.data['img_name'].values]
        # Read only image files in following format
        self.imgs = [i  for i in self.imgs if os.path.splitext(i)[1].lower() in (".mhd", ".mha", ".dcm", ".png", ".jpg", ".jpeg")]   

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", str(self.imgs[idx]))
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
     
        img = Image.fromarray((np.asarray(img)/np.max(img)))
        nodule_data = self.data[self.data['img_name']==str(self.imgs[idx])]
        num_objs = len(nodule_data)
        boxes = []
        
        if nodule_data['label'].any()==1: # nodule data
            for i in range(num_objs):
                x_min = int(nodule_data.iloc[i]['x'])
                y_min = int(nodule_data.iloc[i]['y'])
                y_max = int(y_min+nodule_data.iloc[i]['height'])
                x_max = int(x_min+nodule_data.iloc[i]['width'])
                boxes.append([x_min, y_min, x_max, y_max])

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            labels = torch.ones((num_objs,), dtype=torch.int64)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        # for non-nodule images
        else:
            boxes = torch.empty([0,4])
            area = torch.tensor([0])
            labels = torch.zeros(0, dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

            
        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        image_name = str(self.imgs[idx])

        return img, target, image_name

    def __len__(self):
        return len(self.imgs)
    
    
    
    