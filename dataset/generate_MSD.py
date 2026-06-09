import SimpleITK as sitk
import os
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import pickle
from collections import OrderedDict
from PIL import Image
import re

def generate_MSD_dataset(data_dir, imgs_save_dir, labels_save_dir):
    
    #task_name = "Task05_Prostate"
    task_name = "Task09_Spleen"
    # task_name = "Task06_Lung"
    # task_name = "Task10_Colon"
    # task_name = "Task04_Hippocampus"
    # task_name = "Task07_Pancreas" 
   # task_name = "Task08_HepaticVessel"
    # task_name = "Task01_BrainTumour"
    
    prefix_imagesTr = os.path.join(data_dir, task_name, 'imagesTr')
    prefix_labelsTr = os.path.join(data_dir, task_name, 'labelsTr')

    image_key_origin = os.listdir(prefix_imagesTr)
    image_key = [file for file in image_key_origin if not file.startswith(".")]
    # print(image_key)
    # print(len(np.unique(image_key)))
    
    print(len(image_key))
    # raise ValueError
    label_key = os.listdir(prefix_labelsTr)
    image_key.sort()
    label_key.sort()
    
    name_str = []
    
    for i in range(0, len(image_key)):
        
        print(f'processing i={i}')
        key  = image_key[i]
        print(key)
        pattern = r'spleen_(\d+).nii.gz'
        result = re.search(pattern, key)
        name_str.append(int(result.group(1)))
        # continue
        image_path = os.path.join(prefix_imagesTr, key)
        label_path = os.path.join(prefix_labelsTr, key)
        
        image = sitk.ReadImage(image_path)  
        label = sitk.ReadImage(label_path)

        image_npy = sitk.GetArrayViewFromImage(image)
        # print(image_npy.shape)
        # continue
        label_npy = sitk.GetArrayViewFromImage(label)
        
        if len(image_npy.shape) == 4:

            print("image_npy shape: {}".format(image_npy.shape))
            print("label_npy shape: {}".format(label_npy.shape))
            
            for j in range(image_npy.shape[1]):
                tmp_image = image_npy[:,j,:,:]
                tmp_label = label_npy[j,:,:]

                # print(tmp_image.shape)
                # print(tmp_label.shape)
                
                
                # image Normalize to 0 - 255
                # tmp_image = tmp_image - tmp_image.min()
                # tmp_image = tmp_image / tmp_image.max() * 255
                # tmp_image  = tmp_image.astype(np.uint8)
                
                # npy格式的image保存
                
                maybe_mkdir_p(os.path.join(imgs_save_dir, task_name, 'images', str(i)))
                save_path_image = os.path.join(imgs_save_dir, task_name, 'images', str(i), 'frame_{}'.format(j))
                np.save(save_path_image, tmp_image)
                
                
                # png格式的label保存
                
                maybe_mkdir_p(os.path.join(labels_save_dir, task_name, 'labels', str(i)))
                Image.fromarray(tmp_label).save(os.path.join(labels_save_dir, task_name, 'labels', str(i), 'frame_{}.png'.format(j)))
                
                
                # 第一通道的图像的color map
                
                # color_map = {0:{0, 0, 0}, 1:{255, 0, 0}, 2:{0, 255, 0}, 3:{0, 0, 255},
                #              4: {255, 255, 0}, 5:{255, 0, 255}, 6:{0, 255, 255}, 7:{64, 64, 64},
                #              8:{128, 128, 128}, 9:{255, 128, 128}, 10:{128, 255, 128}, 11:{128, 128, 255},
                #              12: {255, 255, 128}, 13:{255, 128, 255}, 14:{128, 255, 255}, 15:{192, 192, 192},
                #              16:{255, 192, 192}, 17:{192, 255, 192}, 18:{192, 192, 255}, 19:{255, 255, 192}, 
                #              20:{255, 192, 255}, 21:{192, 255, 255}, 22:{128, 128, 128}, 23:{255, 128, 192},
                #              24:{128, 192, 255}, 25:{128, 255, 192}, 26:{192, 192, 255}, 27:{192, 255, 192}}
                
               

                color_map = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                        [255, 0, 255], [0, 255, 255], [64, 64, 64], [128, 128, 128],  
                        [255, 128, 128], [128, 255, 128], [128, 128, 255], 
                        [255, 255, 128], [255, 128, 255], [128, 255, 255], 
                        [192, 192, 192], [255, 192, 192], [192, 255, 192], 
                        [192, 192, 255], [255, 255, 192], [255, 192, 255], 
                        [192, 255, 255], [128, 128, 128], [255, 128, 192], 
                        [128, 192, 255], [128, 255, 192], [192, 192, 255],
                        [192, 255, 192]])
                
                color_label = np.empty((tmp_label.shape[0], tmp_label.shape[1], 3))
                    
                
                for row in range(0, tmp_label.shape[0]):
                    for col in range(0, tmp_label.shape[1]):
                        # print(tmp_label[row, col])
                        # print(color_map[1])
                        # raise ValueError
                        color_label[row, col, :] = color_map[tmp_label[row, col]]
                    
                
                color_label = color_label.astype(np.uint8)
                maybe_mkdir_p(os.path.join(labels_save_dir, task_name, 'color' ,str(i)))
                Image.fromarray(color_label).save(os.path.join(labels_save_dir, task_name, 'color', str(i), 'frame_{}.png'.format(j)))

        else:
            print(image_npy.shape)
            print(label_npy.shape)

            for j in range(image_npy.shape[0]):
                tmp_image = image_npy[j,:,:]
                tmp_label = label_npy[j,:,:]    
                # image Normalize to 0 - 255
                tmp_image = tmp_image - tmp_image.min()
                tmp_image = tmp_image / tmp_image.max() * 255
                tmp_image  = tmp_image.astype(np.uint8)
                
                
                #label
                # print(np.unique(tmp_label))
                # raise ValueError
                # tmp_label = int(tmp_label)
                tmp_label = tmp_label.round().astype(np.int32)
                
                # png格式的image保存

                
                maybe_mkdir_p(os.path.join(imgs_save_dir, task_name, 'images', str(i)))
                Image.fromarray(tmp_image).save(os.path.join(imgs_save_dir, task_name, 'images', str(i), 'frame_{}.png'.format(j)))
                
                # png格式的label保存
                
                maybe_mkdir_p(os.path.join(labels_save_dir, task_name, 'labels', str(i)))
                Image.fromarray(tmp_label).save(os.path.join(labels_save_dir, task_name, 'labels', str(i), 'frame_{}.png'.format(j)))
                
                
                # color map
                
                # color_map = {0:{0, 0, 0}, 1:{255, 0, 0}, 2:{0, 255, 0}, 3:{0, 0, 255},
                #              4: {255, 255, 0}, 5:{255, 0, 255}, 6:{0, 255, 255}, 7:{64, 64, 64},
                #              8:{128, 128, 128}, 9:{255, 128, 128}, 10:{128, 255, 128}, 11:{128, 128, 255},
                #              12: {255, 255, 128}, 13:{255, 128, 255}, 14:{128, 255, 255}, 15:{192, 192, 192},
                #              16:{255, 192, 192}, 17:{192, 255, 192}, 18:{192, 192, 255}, 19:{255, 255, 192}, 
                #              20:{255, 192, 255}, 21:{192, 255, 255}, 22:{128, 128, 128}, 23:{255, 128, 192},
                #              24:{128, 192, 255}, 25:{128, 255, 192}, 26:{192, 192, 255}, 27:{192, 255, 192}}
                
                color_map = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                        [255, 0, 255], [0, 255, 255], [64, 64, 64], [128, 128, 128],  
                        [255, 128, 128], [128, 255, 128], [128, 128, 255], 
                        [255, 255, 128], [255, 128, 255], [128, 255, 255], 
                        [192, 192, 192], [255, 192, 192], [192, 255, 192], 
                        [192, 192, 255], [255, 255, 192], [255, 192, 255], 
                        [192, 255, 255], [128, 128, 128], [255, 128, 192], 
                        [128, 192, 255], [128, 255, 192], [192, 192, 255],
                        [192, 255, 192]])
                
                color_label = np.empty((tmp_label.shape[0], tmp_label.shape[1], 3))
                    
                
                for row in range(0, tmp_label.shape[0]):
                    for col in range(0, tmp_label.shape[1]):
                        # print(tmp_label[row, col])
                        # print(color_map[1])
                        # raise ValueError
                        color_label[row, col, :] = color_map[tmp_label[row, col]]
                    
                
                color_label = color_label.astype(np.uint8)
                maybe_mkdir_p(os.path.join(labels_save_dir, task_name, 'color' ,str(i)))
                Image.fromarray(color_label).save(os.path.join(labels_save_dir, task_name, 'color', str(i), 'frame_{}.png'.format(j)))
    
    print(name_str)            
             
            
       
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-indir", help="folder where the extracted training data is", type=str, default="/mnt/nasv3/zs/datasets/MSD/3D_volumes/")
    parser.add_argument("-images_outdir", help="folder where to save the data for the 2d network", type=str, default="/mnt/nasv3/zs/datasets/MSD/2D_slices/")
    parser.add_argument("-labels_outdir", help="folder where to save the data for the 2d network", type=str, default="/mnt/nasv3/zs/datasets/MSD/2D_slices/")
    args = parser.parse_args()
    generate_MSD_dataset(args.indir, args.images_outdir, args.labels_outdir)

# python generate_chd.py -indir /afs/crc.nd.edu/user/d/dzeng2/data/chd/raw_image -labeled_outdir /afs/crc.nd.edu/user/d/dzeng2/data/chd/test/supervised -unlabeled_outdir /afs/crc.nd.edu/user/d/dzeng2/data/chd/test/contrastive
