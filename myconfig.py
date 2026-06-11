import argparse
import os

assgin_num = argparse.ArgumentParser()
assgin_num.add_argument("--machine_number", default=0, type=int, help="")


parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--device", type=str, default='None')
parser.add_argument("--num_works", type=int, default=4)
parser.add_argument("--exp_load", type=str, default=None)
parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='results dir')
parser.add_argument('--runs_dir', default='./runs', help='runs dir')

# Data
parser.add_argument("--dataset", type=str, default="chd", help='can be chd, acdc, mmwhs, hvsmr')
parser.add_argument("--data_dir", type=str, default="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/preprocessed_data/2D/")
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument("--enable_few_data", default=False, action='store_true')
parser.add_argument('--sampling_k', type=int, default=10)
parser.add_argument('--data_split_list', type=str, default='data_split_list.txt')

# Model
parser.add_argument("--model_name", type=str, default="UNet2D", help='UNet2D, R2UNet, SimSiam_Unet2D, \
                    BYOL_Unet2D, SimTriplet_Unet2D, PPCL')
parser.add_argument("--initial_filter_size", type=int, default=48)
parser.add_argument("--patch_size", nargs='+', type=int, default=[512, 512])
parser.add_argument("--classes", type=int, default=4)

parser.add_argument("--alpha", type=float, default=0.5)

# Train
parser.add_argument("--experiment_name", type=str, default="contrast_chd_simclr_")
parser.add_argument("--restart", default=False, action='store_true')
# V:\ZengShuang\a1-positional_cl-main\positional_cl-main\results\contrast_acdc_pcl_temp01_thresh035_2022-09-13_15-55-01\model
parser.add_argument("--pretrained_model_path", type=str,
                    default='/mnt/nas/ZengShuang/a1-positional_cl-main/positional_cl-main/pre-train/CHD_model.pth')
# parser.add_argument("--latest_model_path", type=str,
#                     default='')

#method
parser.add_argument("--ssl_method", type=str, default="pixpro", help='simclr, moco, byol, \
                    pixpro, simsiam, simtriplet, pixel_pcl, ppcl')

#optimizer
parser.add_argument("--train_optimizer_name", type=str, default="sgd", help='sgd, lars, lars_simclr, \
                    larc')

#pixpro and pixcontrast

parser.add_argument("--use_pixpro", default=False, action='store_true')

parser.add_argument("--patch_use", default=False, action='store_true')
parser.add_argument("--patch_w", type=int, default=64)
parser.add_argument("--patch_h", type=int, default=64)

parser.add_argument("--pixpro_loss", type=str, default='pix_loss', help='total_loss, pix_loss, instance_loss')

#pretrain
parser.add_argument("--checkpoint_pretrain_interval", type=int, default=1)
parser.add_argument("--checkpoint_pretrain_use", default=False, action="store_true")
parser.add_argument("--path_checkpoint", type=str, default='')
parser.add_argument("--AMP", default=False, action='store_true')
parser.add_argument("--fsdp_use", default=False, action='store_true')
parser.add_argument("--find_unused_parameters", default=False, action='store_true')
parser.add_argument("--pixel_loss", type=str, default='SupConLoss')

#finetune
parser.add_argument("--checkpoint_finetune_interval", type=int, default=10) #10
parser.add_argument("--checkpoint_finetune_use", default=False, action="store_true")
parser.add_argument("--val_interval", type=int, default=10) #20

# result save
parser.add_argument("--result_validation", default=False, action="store_true")
parser.add_argument("--finetune_model_path", type=str, default='')

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--base_lr", type=float, default=0.005)
# parser.add_argument("--warmup_epochs", type=int, default=10)
parser.add_argument("--warmup_lr", type=float, default=0.0)

parser.add_argument("--decay", type=str, default='50-100-150-200')
parser.add_argument("--gamma", type=float, default=0.5)
# parser.add_argument("--optimizer", type=str, default='rmsprop',
#                     choices=('sgd', 'adam', 'rmsprop'))
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
parser.add_argument("--epsilon", type=float, default=1e-8)
parser.add_argument("--do_contrast", default=False, action='store_true')
parser.add_argument("--lr_scheduler", type=str, default='cos')
parser.add_argument("--contrastive_method", type=str, default='simclr', help='simclr, gcl(global contrastive learning), pcl(positional contrastive learning)')

# Loss
parser.add_argument("--temp", type=float, default=0.1)
parser.add_argument("--slice_threshold", type=float, default=0.05)

# distributed data parallel

#parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1), help="number of cpu threads to use during batch generation")
parser.add_argument("--parallel", type=str, default='DP', help='DP, DDP')
parser.add_argument("--num_GPUs", type=int, default=1)
parser.add_argument("--GPU_Name", type=str, default='')
#debug
parser.add_argument("--debug", default=False, action="store_true")
parser.add_argument("--affinity_use", default=False, action="store_true")
#pixel_select
parser.add_argument("--pixel_select", default=False, action='store_true')
parser.add_argument("--kernel_size", type=int, default=1)

#JCL
parser.add_argument("--scale_factor", type=float, default=0.25)
parser.add_argument("--pixel_use", default=False, action="store_true")

#BraTS2018
parser.add_argument("--mode", type=str, default='pretrain', help='pretrain, finetune')

parser.add_argument("--symmetric_net", default=False, action="store_true")
parser.add_argument("--only_encoder", default=False, action="store_true")
parser.add_argument("--block_compare", default=False, action="store_true")
parser.add_argument("--block", type=int, default=2)

#MoCo
parser.add_argument("--moco_k", type=int, default=0)
parser.add_argument("--mlp", default=False, action="store_true")

#SwAV
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")
parser.add_argument("--nmb_prototypes", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--epsilon_sk", type=float, default=0.05)
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")

#random sample for pixel CL
parser.add_argument("--random_sample_pixel", default=False, action="store_true")

parser.add_argument("--unet_deeper", default=False, action="store_true")
#two_stage
parser.add_argument("--two_stage", default=False, action='store_true')

#only pixel
parser.add_argument("--only_pixel", default=False, action='store_true')

# OCTA-500
parser.add_argument("--dataset_mode", type=str, default='', help='6M, 3M, both')

parser.add_argument("--task_name", type=str, default='', help='')
parser.add_argument("--loss_type", type=str, default='CE', help='')

#super pixel
parser.add_argument("--super_pixel", default=False, action='store_true')
parser.add_argument("--n_segments", type=int, default=200, help='number of classes from superpixel')
parser.add_argument("--compactness", type=int, default=5, help='a parameter to control the shape of superpixel')

parser.add_argument("--modal", type=str, default='MR', help='CT, MR')

#DeSD

# Temperature teacher parameters
parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
    help="""Initial value for the teacher temperature: 0.04 works well in most cases.
    Try decreasing it if the training loss does not decrease.""")

parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
    of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
    starting with the default value of 0.04 and increase this slightly if needed.""")

parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
    help='Number of warmup epochs for the teacher temperature.')

parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of the DeSD head output.""")

parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.""")

parser.add_argument('--weight_decay_DeSD', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")

parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
    weight decay. We use a cosine schedule for WD and using a larger decay by
    the end of training improves performance for ViTs.""")

parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")

parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches""")

parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")

parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")

parser.add_argument("--DeSD", default=False, action='store_true')

# JCL new
parser.add_argument("--JCL_new", default=False, action="store_true")
parser.add_argument("--JCL_version2", default=False, action="store_true")

parser.add_argument("--local-rank", type=int, default=os.getenv('LOCAL_RANK', -1))

# SSN
parser.add_argument("--n_spix", type=int, default=50)
parser.add_argument("--n_iter", type=int, default=10)
parser.add_argument("--lab", default=False, action="store_true")

parser.add_argument("--medical_modal", type=str, default='CT', help='CT, MRI')


# WCL
parser.add_argument("--label_method", type=str, default='sim', help='sim, position')

# early stop
parser.add_argument("--early_stop_save_model", default=False, action="store_true")

parser.add_argument("--img_size", type=int, default=512)


# GPS-CL
parser.add_argument("--reduce_memory_mode", type=str, default='block', help='block, stride')
parser.add_argument("--block_size", type=int, default=8)
parser.add_argument("--stride", type=int, default=4)

# DiRA
parser.add_argument("--DiRA_mode", type=str, default='', help='dira, dir')

# Superpixel Method
parser.add_argument("--sp_method", type=str, default='SLIC', help='')
parser.add_argument("--degraded", default=False, action="store_true")
parser.add_argument("--weighted_superCL", default=False, action="store_true")
parser.add_argument('--lambda_sp_intra', type=float, default=1.0)
parser.add_argument('--lambda_wcl', type=float, default=1.0)
parser.add_argument("--sp_number", type=int, default=0, help='')

parser.add_argument("--pixel_head", default=False, action="store_true")
parser.add_argument("--rebuttal", default=False, action="store_true")

# model complexity
parser.add_argument("--model_complexity", default=False, action="store_true")

# SDGCL
parser.add_argument("--level_num", type=int, default=1, help='')
parser.add_argument("--is_cluster_merge", default=False, action="store_true")
parser.add_argument("--new_clusters", type=int, default=0, help='')
parser.add_argument('--level_1', type=float, default=0.0)
parser.add_argument('--level_2', type=float, default=0.0)
parser.add_argument('--level_3', type=float, default=0.0)

parser.add_argument('--ratio_merge', type=float, default=0.0)


def save_args(obj, defaults, kwargs):
    for k, v in defaults.iteritems():
        if k in kwargs: v = kwargs[k]
        setattr(obj, k, v)

def get_config():
    config = parser.parse_args()
    config.data_dir = os.path.expanduser(config.data_dir)
    config.patch_size = tuple(config.patch_size)
    return config




# def get_config():
#     config = parser.parse_args()
#     add_local_rank_argument(parser=parser, number=)
#     config.data_dir = os.path.expanduser(config.data_dir)
#     config.patch_size = tuple(config.patch_size)
#     return config
