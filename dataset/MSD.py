import pickle
import numpy as np
import torch
import os
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from torch.utils.data.dataset import Dataset
from random import choice
from .utils import *
from glob import glob
from PIL import Image

class MSD(Dataset):
    def __init__(self, keys, purpose, args):
        self.data_dir = args.data_dir
        self.patch_size = args.patch_size
        self.purpose = purpose
        self.classes = args.classes
        self.do_contrast = args.do_contrast
        self.imgs = []
        self.labels = []
        self.slice_position = []
        self.partition = []
        self.do_contrast = args.do_contrast
        self.task_name = args.task_name
                                                                        
                                                                           
        
        if self.do_contrast:
            raise ValueError
            pass
                              
                                                                                                 
                                                  

                                 
                                          
                                    
                
                                             
                                   
                
                                                     
                                                                           
                                                                            
                                                  
                                                 
                                              
                           
                                          
                                                                         
        
        else:

            for key in keys:
                    
                prefix_img = os.path.join(self.data_dir, self.task_name, 'images', '{}'.format(key))
                img_frames = os.listdir(prefix_img)
                img_frames.sort()
                
                                   
                                                                 
               
                
                      
                prefix_label = os.path.join(self.data_dir, self.task_name, 'labels', '{}'.format(key))
                label_frames = os.listdir(prefix_label)
                label_frames.sort()
                
                                   
                                                                     
                
                
                
                                             
                                                             
                               
                                        
                                  
                
                
                
                for i in range(0, len(img_frames)):
                    self.imgs.append(os.path.join(prefix_img, img_frames[i]))
                    self.labels.append(os.path.join(prefix_label, label_frames[i]))
                    
                    
                    self.slice_position.append(float(i+1)/len(img_frames))
                    part = len(img_frames) / 4.0
                    if part - int(part) >= 0.5:
                        part = int(part + 1)
                    else:
                        part = int(part)
                    self.partition.append(max(0,min(int(i//part),3)+1))
            

    def __getitem__(self, index):
        if self.do_contrast:
            image = Image.open(self.imgs[index])
            image = np.array(image).astype(np.float32)
            
            img1, img2 = self.prepare_contrast(image)
            
            return img1, img2, self.slice_position[index], self.partition[index]
        else:
            if self.task_name in ['Task05_Prostate','Task01_BrainTumour']:
                image = np.load(self.imgs[index]).astype(np.float32)
            else:
                image = Image.open(self.imgs[index])
                image = np.array(image).astype(np.float32)
            label = Image.open(self.labels[index])
            label = np.array(label).astype(np.float32)
            
                          
                                     
                              
            
                             
                                   
            if self.task_name == 'Task05_Prostate':
                img1, _ = self.prepare_supervised(image[0,:,:], label)
                img2, label = self.prepare_supervised(image[1,:,:], label)
                img = np.concatenate((img1, img2), axis=0)
            elif self.task_name == 'Task01_BrainTumour':
                img1, _ = self.prepare_supervised(image[0,:,:], label)
                img2, _ = self.prepare_supervised(image[1,:,:], label)
                img3, _ = self.prepare_supervised(image[2,:,:], label)
                img4, label = self.prepare_supervised(image[3,:,:], label)
                img = np.concatenate((img1, img2, img3, img4), axis=0)
            else:
                img, label = self.prepare_supervised(image, label)
            return img, label


    def  __len__(self):
        return len(self.imgs)
    
    def prepare_supervised(self, img, label):
        if self.purpose == 'train':
                       
            img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
            label, _  = pad_and_or_crop(label, self.patch_size, mode='fixed', coords=coord)
                                                                                                                     
            data_dict = {'data':img[None, None], 'seg':label[None, None]}
            tr_transforms = []
            tr_transforms.append(MirrorTransform((0, 1)))
            tr_transforms.append(RndTransform(SpatialTransform(self.patch_size, list(np.array(self.patch_size)//2),
                                                            True, (100., 350.), (14., 17.),
                                                            True, (0, 2.*np.pi), (-0.000001, 0.00001), (-0.000001, 0.00001),
                                                            True, (0.7, 1.3), 'constant', 0, 3, 'constant', 0, 0,
                                                            random_crop=False), prob=0.67, alternative_transform=RandomCropTransform(self.patch_size)))

            train_transform = Compose(tr_transforms)
            data_dict = train_transform(**data_dict)
            img = data_dict.get('data')[0]
            label = data_dict.get('seg')[0]
            return img, label
        else:
                          

            img, coord = pad_and_or_crop(img, self.patch_size, mode='centre')
            label, _  = pad_and_or_crop(label, self.patch_size, mode='fixed', coords=coord)
            return img[None], label[None]
    def prepare_contrast(self, img):
                      
        img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
                                                                                                                 
                                            
        data_dict = {'data':img[None, None]}
        tr_transforms = []
        tr_transforms.append(MirrorTransform((0, 1)))
        tr_transforms.append(RndTransform(SpatialTransform(self.patch_size, list(np.array(self.patch_size)//2),
                                                            True, (100., 350.), (14., 17.),
                                                            True, (0, 2.*np.pi), (-0.000001, 0.00001), (-0.000001, 0.00001),
                                                            True, (0.7, 1.3), 'constant', 0, 3, 'constant', 0, 0,
                                                            random_crop=False), prob=0.67, alternative_transform=RandomCropTransform(self.patch_size)))

        train_transform = Compose(tr_transforms)
        data_dict1 = train_transform(**data_dict)
        img1 = data_dict1.get('data')[0]
        data_dict2 = train_transform(**data_dict)
        img2 = data_dict2.get('data')[0]

                                              
                                              
                      
        return img1, img2