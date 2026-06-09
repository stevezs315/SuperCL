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
import pydicom

class CHAOS(Dataset):
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
        self.modal = args.modal
        # self.imgs = glob(os.path.join(self.data_dir, 'imgs', '*.jpg'))
        # self.labels = glob(os.path.join(self.data_dir, 'masks', '*.png'))

        for key in keys:
            if self.modal == 'MR':
                img_frames = os.listdir(os.path.join(self.data_dir, 'MR', '{}'.format(key), 'T2SPIR', 'DICOM_anon'))
                label_frames = os.listdir(os.path.join(self.data_dir, 'MR', '{}'.format(key), 'T2SPIR', 'Ground'))
            
            elif self.modal == 'CT':
                img_frames = os.listdir(os.path.join(self.data_dir, 'CT', '{}'.format(key), 'DICOM_anon'))
                label_frames = os.listdir(os.path.join(self.data_dir, 'CT', '{}'.format(key), 'Ground'))
            
            img_frames.sort()
            label_frames.sort()
            # assert len(img_frames) == len(label_frames)
            for i in range(0, len(img_frames)): #len(img_frames)
                # print(img_frames[i])
                # print(label_frames[i])
                # prefix1 = img_frames[i].split(".")[0]
                # prefix2 = label_frames[i].split(".")[0]
                # assert prefix1 == prefix2, f"Prefix mismatch: '{prefix1}' does not match '{prefix2}'."
                self.imgs.append(os.path.join(self.data_dir, 'MR', '{}'.format(key), 'T2SPIR', 'DICOM_anon', img_frames[i]))
                self.labels.append(os.path.join(self.data_dir, 'MR', '{}'.format(key), 'T2SPIR', 'Ground', label_frames[i]))
                
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
            # image = Image.open(self.imgs[index])
            # image = np.array(image).astype(np.float32)
            if self.modal == 'MR':
                ds = pydicom.read_file(self.imgs[index])
                img = ds.pixel_array
                image = (img / img.max()).astype(np.float32)
                # print(np.max(img))
                # print(np.min(img))
                # raise ValueError

                label = Image.open(self.labels[index])
                
                label = np.array(label)
                
                label_new = np.zeros(label.shape)
                #liver
                label_new[label == 63] = 1
                #right kidney
                label_new[label == 126] = 2
                #left kidney
                label_new[label == 189] = 3
                #spleen
                label_new[label == 252] = 4
                
                label_new = label_new.astype(np.float32)
            #一定要看看数据有没有归一化！！！
            img, label = self.prepare_supervised(image, label_new)
            return img, label


    def  __len__(self):
        return len(self.imgs)
    
    def prepare_supervised(self, img, label):
        if self.purpose == 'train':
            # pad image
            img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
            label, _  = pad_and_or_crop(label, self.patch_size, mode='fixed', coords=coord)
            # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
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
            # resize image

            img, coord = pad_and_or_crop(img, self.patch_size, mode='centre')
            label, _  = pad_and_or_crop(label, self.patch_size, mode='fixed', coords=coord)
            return img[None], label[None]
    def prepare_contrast(self, img):
        # resize image
        img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
        # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
        # print("img :{}".format(img.shape))
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

        # print("img1 :{}".format(img1.shape))
        # print("img2 :{}".format(img2.shape))
        # breakpoint()
        return img1, img2