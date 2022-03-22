from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import torch

class CustomImageDataset(Dataset):
    def __init__(self,path):
        dir_list=os.listdir(path)
        self.full_path_list=[]
        for dir in dir_list:
            tmp_full_path_list=[]
            for tail in os.listdir(os.path.join(path,dir)):
                tmp_full_path_list.append(os.path.join(path,dir,tail))
            self.full_path_list.extend(tmp_full_path_list)
    def __len__(self):
        return len(self.full_path_list)
    def __getitem__(self,idx):
        path_idx=self.full_path_list[idx]
        img=Image.open(path_idx)
        img=transforms.ToTensor()(img)
        label=torch.tensor(int(path_idx.split('/')[-2])).long()
        return img,label


