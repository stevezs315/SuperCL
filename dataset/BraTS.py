import math
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as F_transforms
# from torchvision.transforms.functional import InterpolationMode
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from torchvision import transforms

def padding(img, dst_w, dst_h):
    width, height = F_transforms.get_image_size(img)
    pad_left = pad_right = pad_top = pad_bottom = 0
    if width < dst_w:
        pad_left = (dst_w-width)//2
        pad_right = math.ceil((dst_w-width)/2)
        # pad the height if needed
    if height < dst_h:
        pad_top = (dst_h - height) // 2
        pad_bottom = math.ceil((dst_h - height) / 2)
    paddings = [pad_left, pad_top, pad_right, pad_bottom]
    img = F_transforms.pad(img, paddings, 0, 'constant')

    # crop the outers
    if width > dst_w or height > dst_h:
        img = F_transforms.resize(img,[dst_h,dst_w],interpolation=InterpolationMode.BICUBIC)
    return img

class BraTS(Dataset):
    def __init__(self, keys, args, purpose='train', data_split=0, dstw=192, dsth=192):

        #----传入参数-----#

        self.data_dir = args.data_dir
        self.mode = args.mode
       
        # self.data_list = glob(os.path.join(self.data_dir, 'HGG', 'HGG*')) + \
        # glob(os.path.join(self.data_dir, 'LGG', 'LGG*'))
        self.files = []
        self.slice_position = []
        self.partition = []
        self.purpose = purpose
        self.width = dstw
        self.height = dsth
        self.img_color = transforms.ColorJitter(brightness=0.2)
        if args.mode == 'finetune':
            for idx in range(0, len(keys)):
                key = keys[idx]
                dir_name = ''
                if idx < data_split:
                    frames = subfiles(os.path.join(self.data_dir, 'HGG', 'patient_{}'.format(key)), False, None, ".npy", True)
                    if len(frames) == 0:
                        print("Not HGG files!")
                        raise ValueError
                    dir_name = 'HGG'
                else:
                    frames = subfiles(os.path.join(self.data_dir, 'LGG', 'patient_{}'.format(key)), False, None, ".npy", True)
                    if len(frames) == 0:
                        print("Not LGG files!")
                        raise ValueError
                    dir_name = 'LGG'
                frames.sort()
                i = 0
                for frame in frames:
                    self.files.append(os.path.join(self.data_dir, dir_name, 'patient_{}'.format(key), frame))
                    i = i + 1
            
        elif self.mode == 'pretrain':
            for idx in range(0, len(keys)):
                key = keys[idx]
                frames = subfiles(os.path.join(self.data_dir, '{}'.format(key)), False, None, ".npy", True)
                frames.sort()
                i = 0
                for frame in frames:
                    self.files.append(os.path.join(self.data_dir, '{}'.format(key), frame))
                    self.slice_position.append(float(i+1)/len(frames))
                    part = len(frames) / 4.0
                    if part - int(part) >= 0.5:
                        part = int(part + 1)
                    else:
                        part = int(part)
                    self.partition.append(max(0,min(int(i//part),3)+1))
                    i = i + 1
            
            # self.files = glob(os.path.join(self.data_dir, 'HGG', 'HGG*')) + \
            #                 glob(os.path.join(self.data_dir, 'LGG', 'LGG*'))
                    


        
    def __len__(self):
        return len(self.files)

    def padding(self,img, dst_w, dst_h):
        width, height = F_transforms.get_image_size(img)
        pad_left = pad_right = pad_top = pad_bottom = 0
        if width < dst_w:
            pad_left = (dst_w-width)//2
            pad_right = math.ceil((dst_w-width)/2)
            # pad the height if needed
        if height < dst_h:
            pad_top = (dst_h - height) // 2
            pad_bottom = math.ceil((dst_h - height) / 2)
        paddings = [pad_left, pad_top, pad_right, pad_bottom]
        img = F_transforms.pad(img, paddings, 0, 'constant')

        # crop the outers
        if width > dst_w or height > dst_h:
            img = F_transforms.resize(img,[dst_h,dst_w],interpolation=InterpolationMode.BICUBIC)
        return img

    def z_score(self, slice):
        r'''
        Using min-max normalization here. input shape: (h,w)
        or 0-1 normalization
        '''
        slice = slice.clone().detach().float()
        slice_nonzero = slice[torch.nonzero(slice)].float()
        if torch.std(slice) == 0 or torch.std(slice_nonzero) == 0:
            return slice
        else:
            # tmp = (slice - torch.mean(slice)) / torch.std(slice)
            tmp = (slice-slice.min())/(slice.max()-slice.min())
            return tmp

    def normalize(self, selected_data):
        r''' z-score normalization
        input: 4,w,h  or 1,w,h
        '''
        z_scored_data = torch.zeros_like(selected_data, dtype=torch.float32)
        for i in range(selected_data.shape[0]):
            slice_scored = self.z_score(selected_data[i])
            z_scored_data[i] = slice_scored
        return z_scored_data
    
    def transform(self, img):
        tr_transforms1 = transforms.Compose([
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=(-20,20),translate=(0.1,0.1),
                                 scale=(0.9,1.1), shear=(-0.2,0.2))])
        
        tr_transforms2 = transforms.Compose([
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=(-20,20),translate=(0.1,0.1),
                                 scale=(0.9,1.1), shear=(-0.2,0.2))])
                    # transforms.ElasticTransform(alpha=720.0, sigma=24.0)])

        img1 = tr_transforms1(img)
        img2 = tr_transforms2(img)
        return img1, img2
    
    def __getitem__(self, index):
        
        data = np.load(self.files[index])
        data = torch.tensor(data)
        data = padding(data, dst_w=self.width, dst_h=self.height)
        # print("data shape: {}".format(data.shape))
        if self.mode == 'pretrain':
            #前四个模态
            image  = data[:4, :].squeeze(1)
            image_norm = self.normalize(image)
            img1, img2 = self.transform(image_norm)

            img1 = self.img_color(img1.unsqueeze(1)).squeeze(1)
            img2 = self.img_color(img2.unsqueeze(1)).squeeze(1)
            # img1 = self.img_color(img1).unsqueeze(1)
            # img2 = self.img_color(img2).unsqueeze(1)
            
            # print("img1 shape: {}".format(img1.shape))
            return img1, img2, self.slice_position[index], self.partition[index]
        elif self.mode == 'finetune':
            if self.purpose == 'train':
                image  = data[:4, :].squeeze(1)
                image_norm = self.normalize(image)
                img, _ = self.transform(image_norm)
                img = self.img_color(img.unsqueeze(1)).squeeze(1)
                label = data[4, :]
                label.long().squeeze(0)
                return img, label
            elif self.purpose == 'val':

                image  = data[:4, :].squeeze(1)
                image_norm = self.normalize(image)
                label = data[4, :]
                
                label.long().squeeze(0)
            
            return image_norm, label 
            

        
if __name__ == '__main__':
    dataset = BraTS(args=None)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)
    for id, data in tqdm(enumerate(dataloader), total=len(dataloader), ncols=80): #ncols进度条的字符宽度
        pass