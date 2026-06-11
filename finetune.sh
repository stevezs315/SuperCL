# acdc finetune
samples=("8" "20") # 8, 20
for sample in "${samples[@]}";
do
echo "sample=${sample}";
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 24895 \
train_supervised.py --device cuda:0 --ssl_method GPSCL \
--pretrained_model_path 'model_pth/SuperCL_CHD.pth' --restart \
--batch_size 5 --epochs 100 \
--data_dir "dataset/acdc/out_labeled/" \
--lr 5e-4 --min_lr 5e-6 --dataset acdc --patch_size 352 352 \
--experiment_name ACDC_CHD_your_experiment_name_"${sample}"_ --save epochs_100_batchsize_5x2GPU_lr_5e-6-5e-4 \
--initial_filter_size 32 --classes 4 --enable_few_data --sampling_k "${sample}" \
--data_split_list data_split_list.txt \
--parallel DDP --checkpoint_finetune_interval 10 --GPU_Name '0,1' \
--model_name 'UNet2D_JCL'
done

# mmwhs finetune
samples=("2" "4") # 2, 4
for sample in "${samples[@]}";
do
    echo "sample=${sample}";
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 21412 \
    train_supervised.py --device cuda:0 --batch_size 5 --epochs 100 --lr 5e-4 --min_lr 5e-6 \
    --dataset mmwhs --data_dir 'dataset/mmwhs' --ssl_method GPSCL \
    --patch_size 256 256 --restart \
    --pretrained_model_path model_pth/SuperCL_CHD.pth \
    --experiment_name MMWHS_CHD_your_experiment_name_"${sample}"_ --save epochs_100_lr_5e-6-5e-4 \
    --initial_filter_size 32 --classes 8 --enable_few_data --sampling_k "${sample}" \
    --data_split_list data_split_list.txt \
    --parallel DDP --GPU_Name '0,1' --model_name 'UNet2D_JCL'
done

# hvsmr finetune
samples=("1" "2") # 1, 2
for sample in "${samples[@]}";
do
    echo "sample=${sample}";

    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 20871 \
    train_supervised.py --device cuda:0 --batch_size 5 --epochs 100 --lr 5e-4 --min_lr 5e-6 \
    --dataset hvsmr --data_dir 'dataset/hvsmr' --ssl_method GPSCL \
    --patch_size 352 352 --restart \
    --experiment_name HVSMR_BraTS_your_experiment_name_"${sample}"_  --save epochs_100_lr_5e-6-5e-4 \
    --initial_filter_size 32 --classes 3 --enable_few_data --sampling_k "${sample}" \
    --data_split_list data_split_list.txt \
    --pretrained_model_path model_pth/SuperCL_BraTS.pth \
    --parallel DDP --GPU_Name '0,1' \
    --model_name 'UNet2D_JCL'
done

# chaos finetune
samples=("2" "4") # 2, 4
for sample in "${samples[@]}";
do
    echo "sample=${sample}";

    CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 21121 \
    train_supervised.py --device cuda:0 --ssl_method JCL \
    --pretrained_model_path "model_pth/SuperCL_BraTS.pth" \
    --restart --batch_size 4 --epochs 100 \
    --data_dir "datasets/CHAOS/Train_Sets/" \
    --lr 5e-4 --min_lr 5e-6 --dataset CHAOS --patch_size 256 256 \
    --experiment_name CHAOS_BraTS_your_experiment_name_"${sample}"_ --save epochs_100_lr_5e-6-5e-4 \
    --initial_filter_size 32 --classes 5 --enable_few_data --sampling_k "${sample}" \
    --data_split_list data_split_list.txt \
    --parallel DDP --checkpoint_finetune_interval 10 --GPU_Name '0,1' \
    --model_name UNet2D_JCL
done

# spleen finetune
samples=("4" "8") # 4, 8
for sample in "${samples[@]}";
do
    echo "sample=${sample}";

    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 20068 \
    train_supervised.py --device cuda:0 --ssl_method GPSCL \
    --pretrained_model_path model_pth/SuperCL_KiTS.pth \
    --restart --batch_size 4 --epochs 100 \
    --data_dir "datasets/MSD/2D_slices/" \
    --lr 5e-4 --min_lr 5e-6 --dataset MSD --patch_size 512 512 \
    --experiment_name Spleen_KiTS_your_experiment_name_"${sample}"_ --save epochs_100_lr_5e-6-5e-4 \
    --initial_filter_size 32 --classes 2 --enable_few_data --sampling_k "${sample}" \
    --data_split_list data_split_list.txt \
    --parallel DDP --checkpoint_finetune_interval 10 --GPU_Name '0,1' \
    --model_name UNet2D_JCL --task_name Task09_Spleen --loss_type 'CE'
done

# ISIC finetune
samples=("208" "519") # 208, 519
for sample in "${samples[@]}";
do
    echo "sample=${sample}";
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 23661 \
    train_supervised.py --device cuda:0 --ssl_method GPSCL \
    --restart --pretrained_model_path model_pth/SuperCL_KiTS.pth \
    --batch_size 5 --epochs 100 \
    --data_dir "datasets/ISIC/" \
    --lr 5e-4 --min_lr 5e-6 --dataset ISIC --patch_size 256 256 \
    --experiment_name ISIC_KiTS_your_experiment_name_"${sample}"_ --save epochs_100_lr_5e-6-5e-4 \
    --initial_filter_size 32 --classes 2 --enable_few_data --sampling_k "${sample}" \
    --data_split_list data_split_list.txt \
    --parallel DDP --checkpoint_finetune_interval 10 --GPU_Name '0,1' \
    --model_name 'UNet2D_JCL' --mode 'finetune'
done

# heart finetune
samples=("2" "4") # 2, 4
for sample in "${samples[@]}";
do
    echo "sample=${sample}";

    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 23379 \
    train_supervised.py --device cuda:0 --ssl_method GPSCL \
    --pretrained_model_path model_pth/SuperCL_BraTS.pth \
    --restart --batch_size 4 --epochs 100 \
    --data_dir "datasets/MSD/2D_slices/" \
    --lr 5e-4 --min_lr 5e-6 --dataset MSD --patch_size 320 320 \
    --experiment_name Heart_BraTS_your_experiment_name_"${sample}"_ --save epochs_100_batchsize_4x2GPU_lr_5e-6-5e-4 \
    --initial_filter_size 32 --classes 2 --enable_few_data --sampling_k "${sample}" \
    --data_split_list data_split_list.txt \
    --parallel DDP --checkpoint_finetune_interval 10 --GPU_Name '0,1' \
    --model_name UNet2D_JCL --task_name Task02_Heart --loss_type 'CE'
done

# Hippocampus finetune

samples=("21" "52") # 21, 52
for sample in "${samples[@]}";
do

    echo "sample=${sample}";
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 22229 \
    train_supervised.py --device cuda:0 --ssl_method GPSCL \
    --pretrained_model_path model_pth/SuperCL_BraTS.pth \
    --restart --batch_size 32 --epochs 100 \
    --data_dir "datasets/MSD/2D_slices/" \
    --lr 5e-4 --min_lr 5e-6 --dataset MSD --patch_size 32 32 \
    --experiment_name Hippocampus_BraTS_your_experiment_name_"${sample}"_ --save epochs_100_batchsize_4x2GPU_lr_5e-6-5e-4 \
    --initial_filter_size 32 --classes 3 --enable_few_data --sampling_k "${sample}" \
    --data_split_list data_split_list.txt \
    --parallel DDP --checkpoint_finetune_interval 10 --GPU_Name '0,1' \
    --model_name UNet2D_JCL --task_name Task04_Hippocampus --loss_type 'CE'
done
