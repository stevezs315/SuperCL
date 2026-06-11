# CHD pretrain
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 21678 \
train_contrast.py --device cuda:0 \
--model_name UNet2D_SCL --ssl_method SuperCL \
--dataset chd --batch_size 16 --checkpoint_pretrain_interval 10 --epochs 100 \
--data_dir "datasets/chd/out_unlabeled/" --do_contrast --lr 0.01 \
--experiment_name CHD_pretrain_your_experiment_name_ --save SuperCL --slice_threshold 0.1 \
--temp 0.1 --patch_size 512 512 --initial_filter_size 32 --classes 512 \
--contrastive_method 'superpixel_pcl' --GPU_Name '0,1 of M7' --scale_factor 0.25 \
--pixel_use --parallel DDP --n_segments 100 --compactness 10 --super_pixel \
--reduce_memory_mode 'sample' --stride 16 --AMP --lambda_sp_intra 1.0 --lambda_wcl 0.5 \

# BraTS pretrain

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 21182 \
train_contrast.py --device cuda:0 \
--model_name UNet2D_SCL --ssl_method SuperCL \
--dataset BraTS --batch_size 32 --checkpoint_pretrain_interval 10 --epochs 100 \
--data_dir "datasets/BraTS_unlabeled/unlabeled" --do_contrast --lr 0.01 \
--experiment_name BraTS_pretrain_your_experiment_name_ --save SuperCL --slice_threshold 0.1 \
--temp 0.1 --patch_size 192 192 --initial_filter_size 32 --classes 512 \
--contrastive_method 'superpixel_pcl' --GPU_Name '0,1 of M7' --scale_factor 0.25 \
--pixel_use --parallel DDP --n_segments 100 --compactness 10 --super_pixel \
--reduce_memory_mode 'sample' --stride 16 --AMP \
--mode pretrain --lambda_sp_intra 1.0 --lambda_wcl 0.5 \

# KiTS pretrain

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 25634 \
train_contrast.py --device cuda:0 \
--model_name UNet2D_SCL --ssl_method SuperCL \
--dataset KiTS --batch_size 16 --checkpoint_pretrain_interval 10 --epochs 100 \
--data_dir "datasets/KITS/" --do_contrast --lr 0.01 \
--experiment_name KiTS_pretrain_your_experiment_name_ --save SuperCL --slice_threshold 0.1 \
--temp 0.1 --patch_size 512 512 --initial_filter_size 32 --classes 512 \
--contrastive_method 'superpixel_pcl' --GPU_Name '0,1 of M7' --scale_factor 0.25 \
--pixel_use --parallel DDP --n_segments 100 --compactness 10 --super_pixel \
--reduce_memory_mode 'sample' --stride 16 --AMP --sp_method 'SLIC' \
--lambda_sp_intra 1.0 --lambda_wcl 0.5 \