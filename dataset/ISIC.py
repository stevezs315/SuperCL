import pickle
import numpy as np
import torch
import os
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform, ResizeTransform 
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from torch.utils.data.dataset import Dataset
from random import choice
from .utils import *
from PIL import Image
                            
from skimage.segmentation import slic
                                                           
                                                           
import torchvision.transforms as transforms
from torchvision.transforms.functional import adjust_gamma
import random

def Normalization_result(data):
    max_num = np.max(data)
    min_num = np.min(data)
                                                        
                         
                         
                             
                               
           
    data_new = (data - min_num)/(max_num - min_num + 1e-8)
                                
    return data_new


class ISIC(Dataset):
    def __init__(self, keys, purpose, args):
        self.data_dir = args.data_dir
        self.patch_size = args.patch_size
        self.purpose = purpose
        self.classes = args.classes
        self.do_contrast = args.do_contrast
        self.imgs = []
        self.labels = []
        self.superpixel = args.super_pixel
        self.JCL_version2 = args.JCL_version2
        self.mode = args.mode
        
        for key in keys:
            if self.mode == 'finetune':
                self.imgs.append(os.path.join(self.data_dir, 'image', 'ISIC_{:07d}.jpg'.format(key)))
                self.labels.append(os.path.join(self.data_dir, 'mask', 'ISIC_{:07d}_segmentation.png'.format(key)))
            else:
                self.imgs.append(os.path.join(self.data_dir, 'image', '{}'.format(key)))
                self.labels.append(os.path.join(self.data_dir, 'mask', '{}'.format(key)))

    def __getitem__(self, index):
        if self.do_contrast:
            image = Image.open(self.imgs[index])
            
            
            image = image.convert('L')
            image = np.array(image).astype(np.float32)
                                
                              
            
            if self.superpixel:
                     
                img1, img2, img3, img4 = self.prepare_contrast_JCL(image)
                
                c,h,w = img1.shape[0], img1.shape[1], img1.shape[2]
                

                SS_map1 = slic(np.stack([img1.squeeze(0), img1.squeeze(0), img1.squeeze(0)], axis=2), n_segments=self.n_segments, compactness=self.compactness)
                SS_map2 = slic(np.stack([img2.squeeze(0), img2.squeeze(0), img2.squeeze(0)], axis=2), n_segments=self.n_segments, compactness=self.compactness)
                
                return img1, img2, img3, img4, SS_map1, SS_map2
            
            
            
            elif self.JCL_version2:
                img1, img2, img3, img4 = self.prepare_contrast_JCL(image)
                return img1, img2, img3, img4, self.slice_position[index], self.partition[index]
            else:
                img1, img2 = self.prepare_contrast(image)         
                return img1, img2
        else:
            image = Image.open(self.imgs[index])
                                                          
                                       

                                                                  
                                
                                    
                                                                               
                                       
                                                                               
                                                                                            
                                                                                            
                                                                                            
                                                 
                                        
                                        
                                        
                                
                                     
                              
                                                                                               
                                                                
                                
                                                                     
            

                                                                                          
                              
                                                                             
                              
                                                         
            image = image.convert('L')
            image = np.array(image).astype(np.float32)
            
                                     
                              
            
            label = Image.open(self.labels[index])

                                                                            
                                       
                                                                             
            
            label = np.array(label).astype(np.float32)
                                     
                              
                             
            label = label / 255.0
            if self.JCL_version2:
                                   
                                  
                img, label = self.prepare_supervised_JCL(image, label)
            
            else:
                                                                        
                                                                        
                                                                            
                                                                  
                img, label = self.prepare_supervised(image, label)
                                  
                                  
                                                           
                                                                                   
                                                            
                                           
                                                                                   
                
                                                             
                                                                                   
                                                            
                                           
                                                                          
                                  
            return img, label


    def  __len__(self):
        return len(self.imgs)
    
    def prepare_supervised_JCL(self, img, label):
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
    
    def prepare_supervised(self, img, label):
        if self.purpose == 'train':
                    
            random_scale = random.random() * 0.4 + 0.8
                                 
                              
            h, w = self.patch_size
            resize = ResizeTransform(target_size= (int(h*random_scale), int(w*random_scale)))
            data_dict_1 = {'data':img[None, None], 'seg':label[None, None]}
            data_dict_1 = resize(**data_dict_1)
            img = data_dict_1.get('data')[0][0]
            label = data_dict_1.get('seg')[0][0]
                              
                                
                                                    
                                   
                                                                               
                                                        
                                       
                                                                               
            
                                                      
                                   
                                                                               
                                                        
                                       
                                                                      
            
                              
                       
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
                          
            random_scale = random.random() * 0.4 + 0.8
                                 
                              
            h, w = self.patch_size
            resize = ResizeTransform(target_size= (int(h*random_scale), int(w*random_scale)))
            data_dict_1 = {'data':img[None, None], 'seg':label[None, None]}
            data_dict_1 = resize(**data_dict_1)
            img = data_dict_1.get('data')[0][0]
            label = data_dict_1.get('seg')[0][0]

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
    
    
    def prepare_contrast_JCL(self, img):
                      
        img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
        image_norm = Normalization_result(img)
        img_new = Image.fromarray((image_norm*255).astype(np.uint8))

                                                                          
        tr_transforms_fix_1 = []
        tr_transforms_fix_2 = []
        tr_transforms_fix_1.append(transforms.ColorJitter(brightness=(0, 1)))
        tr_transforms_fix_1.append(transforms.ColorJitter(contrast=(0.8, 1.2)))
        tr_transforms_fix_1.append(transforms.GaussianBlur(kernel_size=3, sigma=(1,3)))
        tr_transforms_fix_1 = transforms.Compose(tr_transforms_fix_1)

        tr_transforms_fix_2.append(transforms.ColorJitter(brightness=(0, 1)))
        tr_transforms_fix_2.append(transforms.GaussianBlur(kernel_size=3, sigma=(1,3)))
        tr_transforms_fix_2 = transforms.Compose(tr_transforms_fix_2)

        img1 = tr_transforms_fix_1(img_new)
        img2 = tr_transforms_fix_2(img_new)

        
        img1 = Normalization_result(np.array(img1))
        img2 = Normalization_result(np.array(img2))
        
        data_dict1 = {'data':img1[None, None]}
        data_dict2 = {'data':img2[None, None]}

        tr_transforms_div = []
        
                                   
        tr_transforms_div.append(MirrorTransform((0, 1)))
        tr_transforms_div.append(RndTransform(SpatialTransform(self.patch_size, list(np.array(self.patch_size)//2),
                                                            True, (100., 350.), (14., 17.),
                                                            True, (0, 2.*np.pi), (-0.000001, 0.00001), (-0.000001, 0.00001),
                                                            True, (0.7, 1.3), 'constant', 0, 3, 'constant', 0, 0,
                                                            random_crop=False), prob=0.67, alternative_transform=RandomCropTransform(self.patch_size)))

        tr_transforms_div = Compose(tr_transforms_div)
        
        
        data_dict3 = tr_transforms_div(**data_dict1)
        img3 = data_dict3.get('data')[0]
        
        data_dict4 = tr_transforms_div(**data_dict2)
        img4 = data_dict4.get('data')[0]
                                              
        
        return img1[None], img2[None], img3, img4