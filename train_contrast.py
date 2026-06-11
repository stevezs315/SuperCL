import os
import gc
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import join
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.BraTS import BraTS
from dataset.chd import CHD
from dataset.KiTS import KiTS
from experiment_log import PytorchExperimentLogger
from loss.contrast_loss import SampleConLoss, SupConLoss
from lr_scheduler import LR_Scheduler
from myconfig import get_config
from network.unet2d import UNet2D_classification_SCL
from utils import AverageMeter, model_state_dict, save_model_state


@torch.no_grad()
def build_connected_component(dist):
    b = dist.size(0)
    dist = dist - torch.eye(b, device=dist.device) * 2
    x = torch.arange(b, device=dist.device)
    y = torch.topk(dist, 1, dim=1, sorted=False)[1].flatten()
    rx = torch.cat([x, y]).cpu().numpy()
    ry = torch.cat([y, x]).cpu().numpy()
    v = np.ones(rx.shape[0])
    graph = csr_matrix((v, (rx, ry)), shape=(b, b))
    _, labels = connected_components(csgraph=graph, directed=True, return_labels=True)
    labels = torch.tensor(labels)
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)
    return mask


def is_main_process(args):
    return (
        args.parallel == "DP"
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


def save_pretrain_checkpoint(model, optimizer, criterion, train_loss, epoch, args):
    if not is_main_process(args):
        return
    checkpoint = {
        "model_state_dict": model_state_dict(model, args),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "current_lr": optimizer.param_groups[0]["lr"],
        "args": args,
        "contrast_loss": criterion.state_dict(),
    }
    path_checkpoint = os.path.join(
        args.model_result_dir, "checkpoint_epoch{}.pkl".format(epoch + 1)
    )
    torch.save(checkpoint, path_checkpoint)


def superpixel_global_mean_map(feature_map, label_map):
    batch, height, width = feature_map.shape
    flat_features = feature_map.reshape(batch, -1)
    flat_labels = label_map.long().reshape(batch, -1)
    outputs = []
    pixel_count = flat_features.size(1)

    for features, labels in zip(flat_features, flat_labels):
        label_count = int(labels.max().item()) + 1
        sums = features.new_zeros(label_count)
        sums.scatter_add_(0, labels, features)
        outputs.append((sums / pixel_count)[labels].view(height, width))

    return torch.stack(outputs, dim=0)


def main():

    args = get_config()

    if args.save == "":
        args.save = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(args.results_dir, args.experiment_name + args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    logger = PytorchExperimentLogger(save_path, "elog", ShowTerminal=True)
    model_result_dir = join(save_path, "model")

    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir, exist_ok=True)
    args.model_result_dir = model_result_dir

    logger.print("superpixel method: {}".format(args.sp_method))

    if args.parallel == "DP":

        args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(args.device)

    elif args.parallel == "DDP":
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if is_main_process(args):
        logger.print(f"saving to {save_path}")
        writer = SummaryWriter("runs/" + args.experiment_name + args.save)

    if args.seed is not None:
        seed = args.seed
        print("seed: {}".format(seed))
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    if is_main_process(args):
        logger.print("creating model ...")

    if args.dataset in ["BraTS"]:
        model = UNet2D_classification_SCL(
            in_channels=4,
            initial_filter_size=args.initial_filter_size,
            kernel_size=3,
            classes=args.classes,
            do_instancenorm=True,
        )

    else:
        model = UNet2D_classification_SCL(
            in_channels=1,
            initial_filter_size=args.initial_filter_size,
            kernel_size=3,
            classes=args.classes,
            do_instancenorm=True,
        )

    if is_main_process(args):
        logger.print(f"model: {model}")

    start_epoch = 0

    if args.checkpoint_pretrain_use:

        path_checkpoint = args.path_checkpoint
        if is_main_process(args):
            logger.print(f"restart from {path_checkpoint}")

        model_dict = model.state_dict()
        checkpoint = torch.load(path_checkpoint)
        state_dict = checkpoint["model_state_dict"]
        start_epoch = checkpoint["epoch"]
        current_lr = checkpoint["current_lr"]
        print("current_lr: {}".format(current_lr))

        for key, param in list(state_dict.items()):
            if key.startswith("module."):
                state_dict[key[7:]] = param
                state_dict.pop(key)
        state_dict_new = {k: v for k, v in state_dict.items() if k in model_dict.keys()}

        model.load_state_dict(state_dict_new)
        print("model state_dict load sucessfully!")

    nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.device)

    if args.parallel == "DDP":
        num_gpus = torch.cuda.device_count()

        print("use {} gpus!".format(num_gpus))

        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

        num_parameters = sum([l.nelement() for l in model.module.parameters()])
        if is_main_process(args):
            logger.print(f"number of parameters: {num_parameters}")

            logger.print(f"Parameters: {args}")

    if args.dataset == "chd":
        training_keys = os.listdir(os.path.join(args.data_dir, "train"))
        training_keys.sort()
        train_dataset = CHD(keys=training_keys, purpose="train", args=args)

    elif args.dataset == "BraTS":

        training_keys = os.listdir(os.path.join(args.data_dir))
        training_keys.sort()
        train_dataset = BraTS(keys=training_keys, args=args, dstw=192, dsth=192)

    elif args.dataset == "KiTS":
        training_keys = os.listdir(os.path.join(args.data_dir, "imgs"))
        training_keys.sort()
        train_dataset = KiTS(keys=training_keys, purpose="train", args=args)

    pin_memory = "cuda" in str(args.device)
    if args.parallel == "DP":
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_works,
            drop_last=True,
            pin_memory=pin_memory,
        )
    elif args.parallel == "DDP":

        train_sampler = DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_works,
            drop_last=True,
            pin_memory=True,
        )

    criterion = SupConLoss(
        threshold=args.slice_threshold,
        temperature=args.temp,
        contrastive_method=args.contrastive_method,
    ).to(args.device)
    criterion_reduce_memory = None
    if args.reduce_memory_mode == "sample":
        criterion_reduce_memory = SampleConLoss(
            temperature=args.temp, sample_stride=args.stride, contrastive_method="gcl"
        ).to(args.device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5
    )
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader))
    if is_main_process(args):
        logger.print(f"Optimizer_Original: {optimizer}")
        logger.print(f"Scheduler_Original: {scheduler}")

    if args.checkpoint_pretrain_use:
        path_checkpoint = args.path_checkpoint
        if is_main_process(args):
            logger.print(f"restart from {path_checkpoint}")

        checkpoint = torch.load(path_checkpoint, map_location=torch.device("cpu"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_loss = checkpoint["train_loss"]

    scaler = torch.cuda.amp.GradScaler(enabled=args.AMP)
    best_loss = np.inf

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        gc.collect()
        torch.cuda.empty_cache()

        if args.parallel == "DDP":
            train_sampler.set_epoch(epoch)

        train_loss = train_superpixel(
            train_loader,
            model,
            criterion,
            criterion_reduce_memory,
            scaler,
            epoch,
            optimizer,
            scheduler,
            logger,
            args,
        )

        if is_main_process(args):
            logger.print(
                "\n Epoch: {0}\t"
                "Training Loss {train_loss:.4f} \t".format(
                    epoch + 1, train_loss=train_loss
                )
            )

            writer.add_scalar("training_loss", train_loss, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if (epoch + 1) % args.checkpoint_pretrain_interval == 0:
            save_pretrain_checkpoint(
                model, optimizer, criterion, train_loss, epoch, args
            )

        save_model_state(
            model, args, "{}_{}_latest.pth".format(args.dataset, args.ssl_method)
        )

        if is_main_process(args) and train_loss < best_loss:
            best_loss = train_loss
            save_model_state(
                model, args, "{}_{}_best.pth".format(args.dataset, args.ssl_method)
            )

    train_time = time.time() - start_time
    train_time_str = time.strftime("%H:%M:%S", time.gmtime(train_time))
    logger.print("Training time:  {}".format(train_time_str))


def train_superpixel(
    data_loader,
    model,
    criterion,
    criterion_reduce_memory,
    scaler,
    epoch,
    optimizer,
    scheduler,
    logger,
    args,
):
    model.train()
    losses = AverageMeter()
    epoch_context = "Epoch[%d/%d]" % (epoch, args.epochs)

    for batch_idx, tup in tqdm(
        enumerate(data_loader), desc=epoch_context, total=len(data_loader), ncols=90
    ):

        scheduler(optimizer, batch_idx, epoch)

        (
            img1,
            img2,
            img3,
            img4,
            ssmap_original1,
            ssmap_original2,
            slice_position,
            partition,
        ) = tup

        image1_var, image2_var, image3_var, image4_var = [
            torch.nan_to_num(img.float(), nan=0.0).to(args.device, non_blocking=True)
            for img in (img1, img2, img3, img4)
        ]

        bsz = img1.shape[0]

        with torch.cuda.amp.autocast(enabled=args.AMP):
            f1, _ = model(image1_var)
            f2, _ = model(image2_var)

            _, instance_3 = model(image3_var)
            _, instance_4 = model(image4_var)

            fmap_h, fmap_w = f1.shape[2], f1.shape[3]

            features = torch.cat(
                [instance_3.unsqueeze(1), instance_4.unsqueeze(1)], dim=1
            )
            if args.dataset in ["ISIC"]:
                loss_instance = criterion(features)
            else:
                loss_instance = criterion(features, labels=slice_position)

            ssmap_original1_ds = F.interpolate(
                ssmap_original1.unsqueeze(1).float().to(args.device, non_blocking=True),
                size=(fmap_h, fmap_w),
                mode="nearest",
            )
            ssmap_original2_ds = F.interpolate(
                ssmap_original2.unsqueeze(1).float().to(args.device, non_blocking=True),
                size=(fmap_h, fmap_w),
                mode="nearest",
            )
            label_sp_1 = ssmap_original1_ds.squeeze(1)
            label_sp_2 = ssmap_original2_ds.squeeze(1)

            loss_sp_intra = 0

            for bs_index in range(0, bsz):
                if criterion_reduce_memory is None:
                    raise ValueError(
                        "criterion_reduce_memory is required for superpixel intra-image contrast."
                    )
                single_f1 = torch.transpose(f1[bs_index].reshape(f1.shape[1], -1), 0, 1)
                single_sp_label1 = label_sp_1[bs_index].view(-1)
                single_f2 = torch.transpose(f2[bs_index].reshape(f2.shape[1], -1), 0, 1)
                sp_features = torch.cat(
                    [single_f1.unsqueeze(1), single_f2.unsqueeze(1)], dim=1
                )
                loss_sp_intra += criterion_reduce_memory(sp_features, single_sp_label1)

            loss_sp_intra = loss_sp_intra / bsz

            f1_avg = torch.mean(f1, dim=1)
            f2_avg = torch.mean(f2, dim=1)
            f_sp_avg1 = superpixel_global_mean_map(f1_avg, label_sp_1)
            f_sp_avg2 = superpixel_global_mean_map(f2_avg, label_sp_2)

            f_sp_avg1 = f_sp_avg1.view(bsz, -1)
            f_sp_avg2 = f_sp_avg2.view(bsz, -1)

            all_sp_avg = torch.cat([f_sp_avg1, f_sp_avg2], dim=0)

            similarity = F.cosine_similarity(
                all_sp_avg.unsqueeze(1), all_sp_avg.unsqueeze(0), dim=2
            )
            sim_mask = build_connected_component(similarity.cpu()).float()
            batch_indices = torch.arange(bsz)
            sim_mask[batch_indices + bsz, batch_indices] = 1
            sim_mask[batch_indices, batch_indices + bsz] = 1

            loss_wcl = criterion(features, mask=sim_mask)

            loss = (
                loss_instance
                + args.lambda_sp_intra * loss_sp_intra
                + args.lambda_wcl * loss_wcl
            )

        scaler.scale(loss).backward()
        losses.update(loss.item(), bsz)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if (batch_idx + 1) % len(data_loader) == 0 and is_main_process(args):
            logger.print(
                f"epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, lr:{optimizer.param_groups[0]['lr']:.6f}, loss:{losses.avg:.4f}"
            )

    return losses.avg


if __name__ == "__main__":
    main()
