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

from PIL import ImageEnhance   
                          
                                                     
                                                                      
                                                      
                                                                      
                                                   
                                                                
                                                   
                                                                    
    
                  


class CHD(Dataset):

    def __init__(self, keys, purpose, args):
        self.data_dir = args.data_dir
        self.patch_size = args.patch_size
        self.purpose = purpose
        self.classes = args.classes
        self.do_contrast = args.do_contrast
        self.debug = args.debug
        self.files = []
        self.affinity_use = args.affinity_use
        self.ssl_method = args.ssl_method
        self.contrastive_method = args.contrastive_method
        self.superpixel = args.super_pixel
        self.n_segments = args.n_segments
        self.compactness = args.compactness
        self.sp_method = args.sp_method
        self.sp_number = args.sp_number
        self.sp_map = []
        self.sp_map_2 = []
        self.sp_map_3 = []
        self.level_num = args.level_num


        with open(os.path.join(self.data_dir, "mean_std.pkl"), 'rb') as f:
            mean_std = pickle.load(f)
        if self.do_contrast:
                                                                                      
            self.slice_position = []
            self.partition = []
            self.means = []
            self.stds = []
            self.frames = []
            self.keys= []
                       
            for key in keys:
                frames = subfiles(join(self.data_dir, 'train', key), False, None, ".npy", True)

                                                                                                             
                frames.sort()
                                                                
                i = 0
                for frame in frames:
                    self.files.append(join(self.data_dir, 'train', key, frame))
                    self.means.append(mean_std[key]['mean'])
                    self.stds.append(mean_std[key]['std'])
                    self.slice_position.append(float(i+1)/len(frames))
                    self.frames.append(len(frames))
                    self.keys.append(key)
                    part = len(frames) / 4.0
                    if part - int(part) >= 0.5:
                        part = int(part + 1)
                    else:
                        part = int(part)
                    self.partition.append(max(0,min(int(i//part),3)+1))
                    i = i + 1
                
                if self.sp_method in ['FH', 'LSC', 'LNSNet', 'superpixelFCN', 
                                      'superpixelFCN_CHD_pretrain', 'LNSNet_CHD_pretrain_loss_s','LNSNet_CHD_pretrain_no_loss_s']:
                    sp_data_dir = os.path.join('/mnt/nasv3/zs/datasets/CHD2D/', self.sp_method)
                    frames = os.listdir(os.path.join(sp_data_dir, key))
                    frames.sort()
                    for frame in frames:     
                        self.sp_map.append(os.path.join(sp_data_dir, key, frame))
                
                elif self.sp_method in ['SH']:
                    sp_data_dir = os.path.join('/mnt/nasv3/zs/datasets/CHD2D/', self.sp_method)
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
                    



                    


        else:
            self.slice_position = []
            self.partition = []
            self.means = []
            self.stds = []
            self.frames = []
            self.keys= []

                  
            if self.ssl_method in ['LPCL']:
                for key in keys:
                    frames = subfiles(join(self.data_dir, 'train', str(key)), False, None, ".npz", True)
                    frames.sort()
                    i = 0
                    for frame in frames:
                        self.means.append(mean_std[str(key)]['mean'])
                        self.stds.append(mean_std[str(key)]['std'])
                        self.files.append(join(self.data_dir, 'train', str(key), frame))

                              
                        self.slice_position.append(float(i+1)/len(frames))
                        self.frames.append(len(frames))
                        self.keys.append(key)
                        part = len(frames) / 4.0
                        if part - int(part) >= 0.5:
                            part = int(part + 1)
                        else:
                            part = int(part)
                        self.partition.append(max(0,min(int(i//part),3)+1))
                        i = i + 1

                      
            else:
                for key in keys:
                    frames = subfiles(join(self.data_dir, 'train', 'ct_'+str(key)), False, None, ".npz", True)

                    frames.sort()
                    for frame in frames:
                        self.means.append(mean_std['ct_'+str(key)]['mean'])
                        self.stds.append(mean_std['ct_'+str(key)]['std'])
                        self.files.append(join(self.data_dir, 'train', 'ct_'+str(key), frame))

                              
                                                                                                      
            
                               
                                      
                                                                   
                                                                 
                                                                                      

        print(f'dataset length: {len(self.files)}')

    def __getitem__(self, index):
        if self.do_contrast:
            image = np.load(self.files[index]).astype(np.float32)
                              
            image -= self.means[index]
            image /= self.stds[index]
            
                                                           
                           
                                                      
                                   
                                                                               
                                                                    
                                       
                                                                               
            
                                
                                
                                                                               
            
            
            

            

            if self.superpixel:
                
                                                    
                                         

                     
                img1, img2, img3, img4 = self.prepare_contrast_JCL(image)
                
                c,h,w = img1.shape[0], img1.shape[1], img1.shape[2]
                                                            


                                                                                                                                  
                
                                                                  
                                                                                                                               
                                                                                                                               

                if self.sp_method == 'SLIC':
                                       
                        SS_map1 = slic(np.stack([img1.squeeze(0), img1.squeeze(0), img1.squeeze(0)], axis=2), n_segments=self.n_segments, compactness=self.compactness)
                        SS_map2 = slic(np.stack([img2.squeeze(0), img2.squeeze(0), img2.squeeze(0)], axis=2), n_segments=self.n_segments, compactness=self.compactness)
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
                
                if self.debug or self.affinity_use:
                    return image, img1, img2, self.slice_position[index], self.partition[index], self.keys[index], self.frames[index]
                elif self.contrastive_method == 'weighted_pcl':
                    return img1, img2, self.slice_position[index], self.frames[index]
                else:
                    return img1, img2, self.slice_position[index], self.partition[index]
        else:
            all_data = np.load(self.files[index])['data']
            img = all_data[0].astype(np.float32)
            img -= self.means[index]
            img /= self.stds[index]
            label = all_data[1].astype(np.float32)
            
                                                                                                   
                                                    
                                   
                                                                               
                                                                                                  
                                       
                                                                               
            
                                                      
                                   
                                                                               
                                                                                                  
                                       
                                                                      
            
                                                      
                                           
                                                                                                    
                                              
                                                                      
                                    
                                      
                                    
                     
                                                                  
                                  
                                    
                                  
            
                              
            if self.ssl_method in ['LPCL']:

                                                                                              
                img1, img2, img3, img4, label = self.prepare_contrast_LPCL(img, label)

                return img1, img2, img3, img4, label, self.slice_position[index], self.partition[index]

            else:
                img, label = self.prepare_supervised(img, label)
                return img, label
            
                                                  
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
    
    def prepare_supervised_JCL(self, img, label):

        if self.purpose == 'train':
                       
            img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
            label, _  = pad_and_or_crop(label, self.patch_size, mode='fixed', coords=coord)
                                                                  
                                     
                              
                                                                                                                     

            
                        
            image_norm = Normalization_result(img)
            img_new = Image.fromarray((image_norm*255).astype(np.uint8))

                                
                             
            label = Normalization_result(label)
            label_new = Image.fromarray((label*255).astype(np.uint8))
                                                                 

                                                                              
            tr_transforms_fix_1 = []
                                      
            tr_transforms_fix_1.append(transforms.ColorJitter(brightness=(0, 1)))
            tr_transforms_fix_1.append(transforms.ColorJitter(contrast=(0.8, 1.2)))
            tr_transforms_fix_1.append(transforms.GaussianBlur(kernel_size=3, sigma=(1,3)))
            tr_transforms_fix_1 = transforms.Compose(tr_transforms_fix_1)

                                                                                   
                                                                                     
                                                                                             
                                                                           

            img_new = tr_transforms_fix_1(img_new)
            label_new = tr_transforms_fix_1(label_new)
                        
            img_new = Normalization_result(np.array(img_new))
            label_new = Normalization_result(np.array(label_new))
                                                         
                                                             
                              

                                        
            data_dict = {'data':img_new[None, None], 'seg':label_new[None, None]}
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
            label_new = data_dict.get('seg')[0]
            return img, label_new
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
        'nonspatial augmentation'

                                                                           
                                                
                                                                      

                                                                            
                                  
                                  
                                                                               
                                                                                 
                                                                                         
                                                                       

                                                                               
                                                                                 
                                                                                         
                                                                       

                                             
                                             

        
                                                     
                                                     

                                       

    
                                          
                        
                                                                           
                                                                                                                   
                                                

                      
                                                                        
                                         
                                         
                                                               
                                                       
                                                       
                                                        
                                                     
                                                     
                                                  
                                                  
                                                    

        
                                              
                      
                                              
                                
                                
        
                                                                                     
        
                                                                         
                                                                             
                                                              
                                                                

        
                                                        
        
                                     
                                                           
                                                                                                                     
                                                                                             
                                                                                                                              
                                                                                                                   
                                                                                                                                                         

                                                        
                                                     
                                          
                                                     
                                          
        
                                                      
                                          
        
                                                      
                                          
        
        

                                                
                                                
                        
                                       

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
        tr_transforms_fix_2.append(transforms.ColorJitter(contrast=(0.8, 1.2)))
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

    def prepare_contrast_LPCL(self, img, label):
                      
        img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
        label, _  = pad_and_or_crop(label, self.patch_size, mode='fixed', coords=coord)

        image_norm = Normalization_result(img)
        img_new = Image.fromarray((image_norm*255).astype(np.uint8))

                                                                          
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
                                              
        
        return img1[None], img2[None], img3, img4, label


    def  __len__(self):
        return len(self.files)

class CHD_DiRA(Dataset):

    def __init__(self, keys, do_contrast, augment, args):
        self.data_dir = args.data_dir
        self.patch_size = args.patch_size
                                
                                     
        self.do_contrast = do_contrast
        self.files = []
                                           
        self.augment = augment
        with open(os.path.join(self.data_dir, "mean_std.pkl"), 'rb') as f:
            mean_std = pickle.load(f)
        if self.do_contrast:
                                                                                      
            self.slice_position = []
            self.partition = []
            self.means = []
            self.stds = []
            self.frames = []
            self.keys= []
                       
            for key in keys:
                frames = subfiles(join(self.data_dir, 'train', key), False, None, ".npy", True)

                                                                                                             
                frames.sort()
                                                                
                i = 0
                for frame in frames:
                    self.files.append(join(self.data_dir, 'train', key, frame))
                    self.means.append(mean_std[key]['mean'])
                    self.stds.append(mean_std[key]['std'])
                    self.slice_position.append(float(i+1)/len(frames))
                    self.frames.append(len(frames))
                    self.keys.append(key)
                    part = len(frames) / 4.0
                    if part - int(part) >= 0.5:
                        part = int(part + 1)
                    else:
                        part = int(part)
                    self.partition.append(max(0,min(int(i//part),3)+1))
                    i = i + 1
        else:
            self.means = []
            self.stds = []
            for key in keys:
                frames = subfiles(join(self.data_dir, 'train', 'ct_'+str(key)), False, None, ".npz", True)

                frames.sort()
                for frame in frames:
                    self.means.append(mean_std['ct_'+str(key)]['mean'])
                    self.stds.append(mean_std['ct_'+str(key)]['std'])
                    self.files.append(join(self.data_dir, 'train', 'ct_'+str(key), frame))
                              
                                                                                                      
            
                               
                                      
                                                                   
                                                                 
                                                                                      

        print(f'dataset length: {len(self.files)}')

    def __getitem__(self, index):
        if self.do_contrast:
            image = np.load(self.files[index]).astype(np.float32)
                              
            image -= self.means[index]
            image /= self.stds[index]
           
            image_data = Image.fromarray(image).convert('RGB')
            
            
            return self.augment(image_data)
            
  
        else:
            all_data = np.load(self.files[index])['data']
            img = all_data[0].astype(np.float32)
            img -= self.means[index]
            img /= self.stds[index]
            label = all_data[1].astype(np.float32)
            img, label = self.prepare_supervised(img, label)
            return img, label
            
                                                  

    def  __len__(self):
        return len(self.files)

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










def get_split_chd(data_dir, fold, seed=12345):
                                                 
    all_keys = np.arange(0, 50)
    cases = os.listdir(data_dir)
    cases.sort()
    i = 0
    for case in cases:
      all_keys[i] = int(case[-4:])
      i = i + 1
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = kf.split(all_keys)
    for i, (train_idx, test_idx) in enumerate(splits):
        train_keys = all_keys[train_idx]
        test_keys = all_keys[test_idx]
        if i == fold:
            break
    return train_keys, test_keys


class chd_feature_visualization(Dataset):

    def __init__(self, keys, purpose, args):
        self.data_dir = args.data_dir
        self.patch_size = args.patch_size
        self.purpose = purpose
        self.classes = args.classes
        self.do_contrast = args.do_contrast
        self.files = []
        with open(os.path.join(self.data_dir, "mean_std.pkl"), 'rb') as f:
            mean_std = pickle.load(f)
        self.means = []
        self.stds = []

        for key in keys:
            frames = subfiles(join(self.data_dir, 'train', str(key)), False, None, ".npz", True)

            frames.sort()
            for frame in frames:
                self.means.append(mean_std[str(key)]['mean'])
                self.stds.append(mean_std[str(key)]['std'])
                self.files.append(join(self.data_dir, 'train', str(key), frame))

        print(f'dataset length: {len(self.files)}')

    def __getitem__(self, index):

        all_data = np.load(self.files[index])['data']
        img = all_data[0].astype(np.float32)
        img -= self.means[index]
        img /= self.stds[index]
        label = all_data[1].astype(np.float32)

                                                 
                                                     

        img, label = self.prepare_supervised(img, label)
                                                 
                                                     
                      

        return img, label

    def  __len__(self):
        return len(self.files)

    def prepare_supervised(self, img, label):
                                     
                         
                                                                               
                                                                                             
                                                                                                                       
                                                                           
                                
                                                           
                                                                                                                     
                                                                                             
                                                                                                                              
                                                                                                                   
                                                                                                                                                         
         
                                                      
                                                      
                                            
                                             
                               
               
                          
        img, coord = pad_and_or_crop(img, self.patch_size, mode='centre')
        label, _  = pad_and_or_crop(label, self.patch_size, mode='fixed', coords=coord)
        return img[None], label[None]

















if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/afs/crc.nd.edu/user/d/dzeng2/data/chd/preprocessed_without_label/")
    parser.add_argument("--patch_size", type=tuple, default=(512, 512))
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--do_contrast", default=True, action='store_true')
    parser.add_argument("--slice_threshold", type=float, default=0.05)
    args = parser.parse_args()

    train_keys = os.listdir(os.path.join(args.data_dir,'train'))
    train_keys.sort()
    train_dataset = CHD(keys=train_keys, purpose='train', args=args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=30,
                                                    shuffle=True,
                                                    num_workers=8,
                                                    drop_last=False)

    pp = []
    n = 0
    for batch_idx, tup in enumerate(train_dataloader):
        print(f'the {n}th minibatch...')
        img1, img2, slice_position, partition = tup
        batch_size = img1.shape[0]
                                                                            
        slice_position = slice_position.contiguous().view(-1, 1)
        mask = (torch.abs(slice_position.T.repeat(batch_size,1) - slice_position.repeat(1,batch_size)) < args.slice_threshold).float()
                                                    
        for i  in range(mask.shape[0]):
            pp.append(mask[i].sum()-1)
        n = n + 1
        if n > 100:
            break
    pp = np.asarray(pp)
    pp_mean = np.mean(pp)
    pp_std = np.std(pp)
    print(f'mean:{pp_mean}, std:{pp_std}')