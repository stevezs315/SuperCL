import pickle
import numpy as np
import torch
import os
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from batchgenerators.transforms.color_transforms import *
from batchgenerators.transforms.noise_transforms import *
import torchvision.transforms as transforms
from torchvision.transforms.functional import adjust_gamma
from torch.utils.data.dataset import Dataset
from random import choice
from .utils import *
from glob import glob
from PIL import Image
from skimage.segmentation import slic

def Normalization_result(data):
    max_num = np.max(data)
    min_num = np.min(data)
                                                        
                         
                         
                             
                               
           
    data_new = (data - min_num)/(max_num - min_num + 1e-8)
                                
    return data_new


class KiTS(Dataset):
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
        self.JCL_version2 = args.JCL_version2
        self.superpixel = args.super_pixel
        self.n_segments = args.n_segments
        self.compactness = args.compactness
        self.degraded = args.degraded
        self.sp_method = args.sp_method
        self.ssl_method = args.ssl_method
        
        self.sp_number = args.sp_number
        self.sp_map = []
        self.sp_map_2 = []
        self.sp_map_3 = []
        self.level_num = args.level_num
                                                                        
                                                                           

        for key in keys[:5]:
            img_frames = os.listdir(os.path.join(self.data_dir, 'imgs', '{}'.format(key)))
            label_frames = os.listdir(os.path.join(self.data_dir, 'masks', '{}'.format(key)))
            img_frames.sort()
            label_frames.sort()
                                                         
            for i in range(0, len(img_frames)):                 
                                      
                                        
                                                       
                                                         
                                                                                                        
                self.imgs.append(os.path.join(self.data_dir, 'imgs', '{}'.format(key), img_frames[i]))
                self.labels.append(os.path.join(self.data_dir, 'masks', '{}'.format(key), label_frames[i]))
                
                self.slice_position.append(float(i+1)/len(img_frames))
                part = len(img_frames) / 4.0
                if part - int(part) >= 0.5:
                    part = int(part + 1)
                else:
                    part = int(part)
                self.partition.append(max(0,min(int(i//part),3)+1))
            if self.superpixel:
                if self.sp_method in ['SH']:
                    sp_data_dir = os.path.join('/mnt/nasv3/zs/datasets/KiTS2D', self.sp_method)
                    frames = os.listdir(os.path.join(sp_data_dir, key))

                    if self.sp_number == 128:

                        filtered_frames = [frame for frame in frames if '128' in frame]
                        filtered_frames.sort()
                        for frame in filtered_frames:     
                            self.sp_map.append(os.path.join(sp_data_dir, key, frame))

                    if self.sp_number == 64:

                        filtered_frames = [frame for frame in frames if '64' in frame]
                        filtered_frames.sort()
                        for frame in filtered_frames:     
                            self.sp_map.append(os.path.join(sp_data_dir, key, frame))
                        
                        filtered_frames_2 = [frame for frame in frames if '128' in frame]
                        filtered_frames_2.sort()
                        for frame in filtered_frames_2:     
                            self.sp_map_2.append(os.path.join(sp_data_dir, key, frame)) 


                    if self.sp_number == 32:

                        filtered_frames = [frame for frame in frames if '32' in frame]
                        filtered_frames.sort()
                        for frame in filtered_frames:     
                            self.sp_map.append(os.path.join(sp_data_dir, key, frame))

                        filtered_frames_2 = [frame for frame in frames if '64' in frame]
                        filtered_frames_2.sort()
                        for frame in filtered_frames_2:     
                            self.sp_map_2.append(os.path.join(sp_data_dir, key, frame))

                        filtered_frames_3 = [frame for frame in frames if '128' in frame]
                        filtered_frames_3.sort()
                        for frame in filtered_frames_3:     
                            self.sp_map_3.append(os.path.join(sp_data_dir, key, frame))
                    
                    if self.sp_number == 16:

                        filtered_frames = [frame for frame in frames if '16' in frame]
                        filtered_frames.sort()
                        for frame in filtered_frames:     
                            self.sp_map.append(os.path.join(sp_data_dir, key, frame))

                        filtered_frames_2 = [frame for frame in frames if '32' in frame]
                        filtered_frames_2.sort()
                        for frame in filtered_frames_2:     
                            self.sp_map_2.append(os.path.join(sp_data_dir, key, frame))

                        filtered_frames_3 = [frame for frame in frames if '64' in frame]
                        filtered_frames_3.sort()
                        for frame in filtered_frames_3:     
                            self.sp_map_3.append(os.path.join(sp_data_dir, key, frame))
                    
                    if self.sp_number == 8:

                        filtered_frames = [frame for frame in frames if '8' in frame]
                        filtered_frames.sort()
                        for frame in filtered_frames:     
                            self.sp_map.append(os.path.join(sp_data_dir, key, frame))

                        filtered_frames_2 = [frame for frame in frames if '16' in frame]
                        filtered_frames_2.sort()
                        for frame in filtered_frames_2:     
                            self.sp_map_2.append(os.path.join(sp_data_dir, key, frame))

                        filtered_frames_3 = [frame for frame in frames if '32' in frame]
                        filtered_frames_3.sort()
                        for frame in filtered_frames_3:     
                            self.sp_map_3.append(os.path.join(sp_data_dir, key, frame))


    def __getitem__(self, index):
        if self.do_contrast:
            image = Image.open(self.imgs[index])
            image = np.array(image).astype(np.float32)


            if self.superpixel:
                img1, img2, img3, img4 = self.prepare_contrast_JCL(image)

                if self.degraded:
                    return img1, img2, img3, img4, self.slice_position[index], self.partition[index]
                
                else:
                
                    c,h,w = img1.shape[0], img1.shape[1], img1.shape[2]
                                                                

                                      

                                                                                                                                      
                    
                                                                      
                                                                                                                                   
                                                                                                                                   
                    if self.sp_method == 'SLIC':
                        SS_map1 = slic(np.stack([img1.squeeze(0), img1.squeeze(0), img1.squeeze(0)], axis=2), n_segments=self.n_segments, compactness=self.compactness)
                        SS_map2 = slic(np.stack([img2.squeeze(0), img2.squeeze(0), img2.squeeze(0)], axis=2), n_segments=self.n_segments, compactness=self.compactness)
                    elif self.sp_method in ['FH', 'LSC', 'LNSNet', 'superpixelFCN', 
                                            'superpixelFCN_CHD_pretrain', 'LNSNet_CHD_pretrain_loss_s', 'LNSNet_CHD_pretrain_no_loss_s']:
                        
                        SS_map = Image.open(self.sp_map[index])
                        SS_map = np.asarray(SS_map)
                        h,w = SS_map.shape
                        if h != 512 or w != 512:
                            new_shape = [512, 512]
                            SS_map = pad_if_too_small(SS_map, new_shape, pad_value=0)
                                                   
                        SS_map1 = SS_map
                        SS_map2 = SS_map
                    
                    elif self.sp_method in ['SH']:
                        if self.level_num >= 1:
                            SS_map1 = Image.open(self.sp_map[index]).convert('L')
                            SS_map1 = np.asarray(SS_map1)
                            h,w = SS_map1.shape
                            if h != 512 or w != 512:
                                new_shape = [512, 512]
                                SS_map1 = pad_if_too_small(SS_map1, new_shape, pad_value=0)
                            
                                
                                                  
                                                         
                                              


                        if self.level_num >=2:
                            SS_map2 = Image.open(self.sp_map_2[index]).convert('L')
                            SS_map2 = np.asarray(SS_map2)
                            h,w = SS_map2.shape
                            if h != 512 or w != 512:
                                new_shape = [512, 512]
                                SS_map2 = pad_if_too_small(SS_map2, new_shape, pad_value=0)
                            
                                                       
                            
                        if self.level_num >=3:
                            SS_map3 = Image.open(self.sp_map_3[index]).convert('L')
                            SS_map3 = np.asarray(SS_map3)
                            h,w = SS_map3.shape
                            if h != 512 or w != 512:
                                new_shape = [512, 512]
                                SS_map3 = pad_if_too_small(SS_map3, new_shape, pad_value=0)
                    
                    if self.sp_method in ['SH']:
                        if self.level_num == 1:
                                               
                                               
                                               
                                               
                                              
                            return img1, img2, img3, img4, SS_map1, self.slice_position[index], self.partition[index]
                        if self.level_num == 2:
                            return img1, img2, img3, img4, SS_map1, SS_map2, self.slice_position[index], self.partition[index]
                        if self.level_num == 3:
                            return img1, img2, img3, img4, SS_map1, SS_map2, SS_map3, self.slice_position[index], self.partition[index]
                    else:
                        return img1, img2, img3, img4, SS_map1, SS_map2, self.slice_position[index], self.partition[index]

            else:
                img1, img2 = self.prepare_contrast(image)
            
                return img1, img2, self.slice_position[index], self.partition[index]
        else:
            if self.ssl_method in ['LPCL']:
                image = Image.open(self.imgs[index])
                image = np.array(image).astype(np.float32)

                label = Image.open(self.labels[index])
                label = np.array(label).astype(np.float32)
                label = label / 255.0
                label[label > 0.5] = 1
                label[label!=1] = 0
                                         
                                  

                img1, img2, img3, img4, label = self.prepare_contrast_LPCL(image, label)
                return img1, img2, img3, img4, label, self.slice_position[index], self.partition[index]
            
            else:

                image = Image.open(self.imgs[index])
                image = np.array(image).astype(np.float32)
                
                                         
                                  
                
                label = Image.open(self.labels[index])
                label = np.array(label).astype(np.float32)
                                 
                label = label / 255.0
                label[label > 0.5] = 1
                label[label!=1] = 0
                
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
    
    def prepare_contrast_JCL(self, img):
                      
        img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
                                                                                                                 
                                            

                          
                            
                          

        img = Image.fromarray((img).astype(np.uint8))


                                                                          
        tr_transforms_fix_1 = []
        tr_transforms_fix_2 = []
        tr_transforms_fix_1.append(transforms.ColorJitter(brightness=(0, 1)))
        tr_transforms_fix_1.append(transforms.ColorJitter(contrast=(0.8, 1.2)))
        tr_transforms_fix_1.append(transforms.GaussianBlur(kernel_size=3, sigma=(1,3)))
        tr_transforms_fix_1 = transforms.Compose(tr_transforms_fix_1)

        tr_transforms_fix_2.append(transforms.ColorJitter(brightness=(0, 1)))
        tr_transforms_fix_2.append(transforms.ColorJitter(contrast=(0.8, 1.2)))
        tr_transforms_fix_2.append(transforms.GaussianBlur(kernel_size=3, sigma=(1,3)))
        tr_transforms_fix_2 = transforms.Compose(tr_transforms_fix_2)

        img1 = tr_transforms_fix_1(img)
        img2 = tr_transforms_fix_2(img)

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
    
    def prepare_contrast_LPCL(self, img, label):
                      
        img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
        label, _  = pad_and_or_crop(label, self.patch_size, mode='fixed', coords=coord)
                                                                                                                 
                                            

                          
                            
                          

        img = Image.fromarray((img).astype(np.uint8))


                                                                          
        tr_transforms_fix_1 = []
        tr_transforms_fix_2 = []
        tr_transforms_fix_1.append(transforms.ColorJitter(brightness=(0, 1)))
        tr_transforms_fix_1.append(transforms.ColorJitter(contrast=(0.8, 1.2)))
        tr_transforms_fix_1.append(transforms.GaussianBlur(kernel_size=3, sigma=(1,3)))
        tr_transforms_fix_1 = transforms.Compose(tr_transforms_fix_1)

        tr_transforms_fix_2.append(transforms.ColorJitter(brightness=(0, 1)))
        tr_transforms_fix_2.append(transforms.ColorJitter(contrast=(0.8, 1.2)))
        tr_transforms_fix_2.append(transforms.GaussianBlur(kernel_size=3, sigma=(1,3)))
        tr_transforms_fix_2 = transforms.Compose(tr_transforms_fix_2)

        img1 = tr_transforms_fix_1(img)
        img2 = tr_transforms_fix_2(img)

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

        
        return img1[None], img2[None], img3, img4, label