import os
import re
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from dataset.acdc import ACDC
from dataset.CHAOS import CHAOS
from dataset.hvsmr import HVSMR
from dataset.ISIC import ISIC
from dataset.MSD import MSD
from dataset.mmwhs import MMWHS
from experiment_log import PytorchExperimentLogger
from lr_scheduler import LR_Scheduler
from metrics import Dice_Cal, SegmentationMetric
from myconfig import get_config
from network.unet2d import UNet2D
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from medpy import metric
from utils import AverageMeter, model_state_dict, save_model_state


SUPERVISED_DATASETS = {
    "CHAOS": CHAOS,
    "ISIC": ISIC,
    "MSD": MSD,
    "mmwhs": MMWHS,
    "acdc": ACDC,
    "hvsmr": HVSMR,
}
MULTICLASS_LOSSES = {"CE", "DiceCE"}


def calculate_surface_metrics(pred, gt):
    return metric.binary.hd95(pred, gt), metric.binary.asd(pred, gt)


def DiceCE_loss(pred, target):
    smooth = 1.0
    pred_ = torch.max(pred, dim=1)[0]
    intersection = torch.sum(pred_ * target, dim=(1, 2))
    union = torch.sum(pred_ + target, dim=(1, 2))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()
    ce_loss = F.cross_entropy(pred, target)
    return 0.1 * dice_loss + ce_loss


def is_main_process(args):
    return (
        args.parallel == "DP"
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


def save_training_checkpoint(model, optimizer, epoch, best_dice, args):
    if not is_main_process(args):
        return
    checkpoint = {
        "model_state_dict": model_state_dict(model, args),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch + 1,
        "best_dice": best_dice,
    }
    path_checkpoint = os.path.join(
        args.model_result_dir, "checkpoint_epoch{}.pkl".format(epoch + 1)
    )
    torch.save(checkpoint, path_checkpoint)


def _adapt_input_channels(image_var, args):
    if args.dataset in ["CANDI", "CHAOS", "AMOS", "hvsmr"]:
        return image_var.repeat(1, 4, 1, 1)

    if args.task_name in ["Task04_Hippocampus", "Task02_Heart"]:
        return image_var.repeat(1, 4, 1, 1)

    return image_var


def _extract_int_list(text):
    return [int(item) for item in re.findall(r"-?\d+", text)]


def _resolve_split_list_path(path):
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _parse_data_split_list(path):
    split_path = _resolve_split_list_path(path)
    if not os.path.exists(split_path):
        raise FileNotFoundError("data split list not found: {}".format(split_path))

    split_data = {}
    section_name = None
    section_lines = []

    def flush_section():
        if section_name is None:
            return
        section_text = "\n".join(section_lines)
        samples = {}
        for match in re.finditer(
            r"sample\s+(\d+)\s+train_keys?\s*:\s*\[(.*?)\]", section_text, re.I | re.S
        ):
            samples[int(match.group(1))] = _extract_int_list(match.group(2))

        test_match = re.search(r"test\s*:\s*\[(.*?)\]", section_text, re.I | re.S)
        test_keys = _extract_int_list(test_match.group(1)) if test_match else []
        split_data[section_name.lower()] = {
            "samples": samples,
            "test": test_keys,
        }

    with open(split_path, "r") as split_file:
        for raw_line in split_file:
            line = raw_line.strip()
            if line.startswith("#"):
                flush_section()
                section_name = line[1:].strip()
                section_lines = []
            else:
                section_lines.append(raw_line.rstrip("\n"))
    flush_section()
    return split_data


def _split_list_section_name(args):
    dataset_name = args.dataset.lower()
    dataset_sections = {
        "acdc": "acdc",
        "mmwhs": "mmwhs",
        "hvsmr": "hvsmr",
        "chaos": "chaos",
        "isic": "isic",
    }
    if dataset_name == "msd":
        task_sections = {
            "Task09_Spleen": "spleen",
            "Task02_Heart": "heart",
            "Task04_Hippocampus": "hippocampus",
        }
        return task_sections.get(args.task_name)
    return dataset_sections.get(dataset_name)


def _load_fixed_split(args):
    section_name = _split_list_section_name(args)
    if section_name is None:
        return None

    split_data = _parse_data_split_list(args.data_split_list)
    if section_name not in split_data:
        raise ValueError(
            "No split section '{}' found in {}".format(
                section_name, args.data_split_list
            )
        )

    section = split_data[section_name]
    sample = int(args.sampling_k)
    if sample not in section["samples"]:
        available = sorted(section["samples"].keys())
        raise ValueError(
            "No sample {} train_keys found for '{}'. Available samples: {}".format(
                sample, section_name, available
            )
        )
    if len(section["test"]) == 0:
        raise ValueError(
            "No test keys found for '{}' in {}".format(
                section_name, args.data_split_list
            )
        )

    return section["samples"][sample], section["test"], section_name


def build_supervised_datasets(args, train_keys, val_keys, logger):
    dataset_class = SUPERVISED_DATASETS.get(args.dataset)
    if dataset_class is None:
        raise ValueError(
            "Dataset '{}' is not configured for supervised fixed-list training.".format(
                args.dataset
            )
        )

    if is_main_process(args):
        logger.print(f"train_keys:{len(train_keys), train_keys}")
        logger.print(f"val_keys:{len(val_keys), val_keys}")

    train_dataset = dataset_class(keys=train_keys, purpose="train", args=args)
    validate_dataset = dataset_class(keys=val_keys, purpose="val", args=args)

    if is_main_process(args) and hasattr(train_dataset, "data_dir"):
        logger.print("training data dir " + train_dataset.data_dir)

    return train_dataset, validate_dataset


def run(writer, args, checkpoint):

    split_result_dir = os.path.join(args.save_path, "list_split")
    maybe_mkdir_p(split_result_dir)
    logger = PytorchExperimentLogger(split_result_dir, "elog", ShowTerminal=True)
    if is_main_process(args):
        logger.print(f"Method: {args.ssl_method}")
        logger.print(f"Sample: {args.sampling_k}")

    if is_main_process(args):
        logger.print(f"Parameters: {args}")

    seed = args.seed
    torch.manual_seed(seed)
    if "cuda" in str(args.device):
        torch.cuda.manual_seed_all(seed)

    if is_main_process(args):
        logger.print("starting training with fixed data split list ...")
    model_result_dir = join(split_result_dir, "model")
    maybe_mkdir_p(model_result_dir)
    args.model_result_dir = model_result_dir

    logger.print("creating model ...")
    if args.model_name == "UNet2D_JCL":

        if args.dataset in ["acdc", "mmwhs", "MSD", "ISIC"]:
            if args.task_name in ["Task04_Hippocampus", "Task02_Heart"]:
                model = UNet2D(
                    in_channels=4,
                    initial_filter_size=args.initial_filter_size,
                    kernel_size=3,
                    classes=args.classes,
                    do_instancenorm=True,
                )
            else:
                model = UNet2D(
                    in_channels=1,
                    initial_filter_size=args.initial_filter_size,
                    kernel_size=3,
                    classes=args.classes,
                    do_instancenorm=True,
                )

        elif args.dataset in ["CHAOS", "hvsmr"]:
            model = UNet2D(
                in_channels=4,
                initial_filter_size=args.initial_filter_size,
                kernel_size=3,
                classes=args.classes,
                do_instancenorm=True,
            )

    print("model: {}".format(model))

    if not args.checkpoint_finetune_use:
        if args.restart:

            if args.checkpoint_pretrain_use:

                print("Wrong Place!")
                raise ValueError

            else:

                logger.print("loading from saved model " + args.pretrained_model_path)
                dict = torch.load(
                    args.pretrained_model_path,
                    map_location=lambda storage, loc: storage,
                )

                save_model = dict["net"]
                for key, param in list(save_model.items()):
                    if key.startswith("module."):
                        save_model[key[7:]] = param
                        save_model.pop(key)

                model_dict = model.state_dict()

                state_dict = {k: v for k, v in save_model.items() if "encoder" in k}

                logger.print("state_dict :{}".format(state_dict.keys()))
                print("\n")
                logger.print("model_dict: {}".format(model_dict.keys()))

                for name, param in state_dict.items():
                    if name in model_dict.keys():
                        if param.shape == model_dict[name].shape:
                            model.state_dict()[name].copy_(param)
                            logger.print(f"{name} load successfully!")
                        else:
                            logger.print(
                                f"Ignoring parameter '{name}' due to shape mismatch."
                            )

    if args.result_validation:
        logger.print(
            "loading from saved model "
            + os.path.join(args.finetune_model_path, "list_split", "model/latest.pth")
        )
        if args.finetune_model_path == "":
            print("Please write the correct path !")
            breakpoint()
        else:
            dict = torch.load(
                os.path.join(
                    args.finetune_model_path, "list_split", "model/latest.pth"
                ),
                map_location=lambda storage, loc: storage,
            )
            save_model = dict["net"]
            model.load_state_dict(save_model)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    if is_main_process(args):
        logger.print(f"number of parameters: {num_parameters}")

    fixed_split = _load_fixed_split(args)
    if fixed_split is None:
        raise ValueError(
            "Dataset/task is not configured in {}. Please add its train/test lists first.".format(
                args.data_split_list
            )
        )
    fixed_train_keys, fixed_val_keys, fixed_split_name = fixed_split
    if is_main_process(args):
        logger.print(
            "using fixed split list '{}' from {} with sample {}".format(
                fixed_split_name, args.data_split_list, args.sampling_k
            )
        )

    train_dataset, validate_dataset = build_supervised_datasets(
        args, fixed_train_keys, fixed_val_keys, logger
    )

    pin_memory = "cuda" in str(args.device)
    if args.parallel == "DP":
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_works,
            drop_last=False,
            pin_memory=pin_memory,
        )
    elif args.parallel == "DDP":
        train_sampler = DistributedSampler(train_dataset)
        if args.model_name in ["deeplabv3plus_JCL"]:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                sampler=train_sampler,
                batch_size=args.batch_size,
                num_workers=args.num_works,
                drop_last=True,
                pin_memory=pin_memory,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                sampler=train_sampler,
                batch_size=args.batch_size,
                num_workers=args.num_works,
                drop_last=False,
                pin_memory=pin_memory,
            )
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_works,
        drop_last=False,
        pin_memory=pin_memory,
    )

    if args.loss_type == "CE":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_type == "BCE":
        criterion = torch.nn.BCELoss()

    elif args.loss_type == "DiceCE":
        criterion = DiceCE_loss

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )
    scheduler = LR_Scheduler(
        args.lr_scheduler, args.lr, args.epochs, len(train_loader), min_lr=args.min_lr
    )
    best_dice = 0
    start_epoch = 0
    if args.checkpoint_finetune_use:

        best_dice = checkpoint["best_dice"]
        start_epoch = checkpoint["epoch"]
        model_state_dict = checkpoint["model_state_dict"]
        optimizer_state_dict = checkpoint["optimizer_state_dict"]

        for key, param in list(model_state_dict.items()):
            if key.startswith("module."):
                model_state_dict[key[7:]] = param
                model_state_dict.pop(key)

        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    model.to(args.device)

    num_gpus = torch.cuda.device_count()

    print("use {} gpus!".format(num_gpus))
    if args.parallel == "DDP":
        if args.model_name in ["deeplabv3plus_JCL", "R2UNet_JCL", "UNet2D_JCL"]:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False,
            )
        else:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )
    elif args.parallel == "DP":
        model = torch.nn.DataParallel(model)

    for epoch in range(start_epoch, args.epochs):

        if args.parallel == "DDP":
            train_sampler.set_epoch(epoch)

        train_loss, train_dice = train(
            train_loader, model, criterion, epoch, optimizer, scheduler, logger, args
        )

        if is_main_process(args):
            writer.add_scalar("training_loss", train_loss, epoch)
            writer.add_scalar("training_dice", train_dice, epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        if (epoch + 1) % args.checkpoint_finetune_interval == 0:
            save_training_checkpoint(model, optimizer, epoch, best_dice, args)

        if (epoch + 1) % args.val_interval == 0:
            val_dice = validate(validate_loader, model, epoch, logger, args)
            if is_main_process(args):
                logger.print(
                    "Epoch: {0}\t"
                    "Training Loss {train_loss:.4f} \t"
                    "Validation Dice {val_dice:.4f} \t".format(
                        epoch, train_loss=train_loss, val_dice=val_dice
                    )
                )

                if best_dice < val_dice:
                    best_dice = val_dice
                    save_model_state(model, args, "best.pth")
                writer.add_scalar("validate_dice", val_dice, epoch)
                writer.add_scalar("best_dice", best_dice, epoch)
                save_model_state(model, args, "latest.pth")


def train(data_loader, model, criterion, epoch, optimizer, scheduler, logger, args):
    model.train()
    metric_val = SegmentationMetric(args.classes)
    metric_val.reset()
    losses = AverageMeter()

    Dice_total = Dice_Cal()
    Dice_total.reset()
    epoch_context = "Epoch[%d/%d]" % (epoch, args.epochs)

    for batch_idx, tup in tqdm(
        enumerate(data_loader), desc=epoch_context, total=len(data_loader), ncols=90
    ):
        img, label = tup

        image_var = img.float().to(args.device, non_blocking=True)
        if args.loss_type in MULTICLASS_LOSSES:
            label = label.long().to(args.device, non_blocking=True)
        elif args.loss_type == "BCE":
            label = label.to(args.device, non_blocking=True)

        scheduler(optimizer, batch_idx, epoch)

        image_var = _adapt_input_channels(image_var, args)

        x_out = model(image_var)

        if args.loss_type in MULTICLASS_LOSSES:
            loss = criterion(x_out, label.squeeze(dim=1))
        elif args.loss_type == "BCE":
            loss = criterion(torch.sigmoid(x_out), label)

        losses.update(loss.item(), image_var.size(0))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        optimizer.step()
        if args.loss_type in MULTICLASS_LOSSES:
            x_out = F.softmax(x_out, dim=1)
            metric_val.update(label.long().squeeze(dim=1), x_out)
            _, _, Dice = metric_val.get()

        elif args.loss_type in ["BCE"]:
            Dice_each = Dice_total.calculate(
                torch.sigmoid(x_out).squeeze(1), label.squeeze(1)
            )
            Dice_total.update(Dice_each)
            Dice = Dice_total.get()

        if is_main_process(args) and (
            (batch_idx + 1) % 50 == 0 or (batch_idx + 1) % len(data_loader) == 0
        ):
            logger.print(
                f"Training epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, lr:{optimizer.param_groups[0]['lr']:.6f}, loss:{losses.avg:.4f}, mean Dice:{Dice:.4f}"
            )

    if args.loss_type in MULTICLASS_LOSSES:
        _, _, mDice = metric_val.get()
    elif args.loss_type in ["BCE"]:
        mDice = Dice_total.get()

    return losses.avg, mDice


def validate(data_loader, model, epoch, logger, args):
    model.eval()
    metric_val = SegmentationMetric(args.classes)
    metric_val.reset()

    hd95_class = torch.zeros((args.classes - 1, len(data_loader)))
    asd_class = torch.zeros((args.classes - 1, len(data_loader)))

    with torch.no_grad():

        for batch_idx, tup in enumerate(data_loader):

            img, label = tup

            image_var = img.float().to(args.device, non_blocking=True)
            image_var = _adapt_input_channels(image_var, args)

            x_out = model(image_var)

            label = label.long().to(args.device, non_blocking=True)
            if args.loss_type in MULTICLASS_LOSSES:
                x_out = F.softmax(x_out, dim=1)

                metric_val.update(label.long().squeeze(dim=1), x_out)
                _, mIoU, mDice = metric_val.get()

                x_out_new = torch.argmax(x_out, dim=1)
                pred_np = x_out_new.detach().cpu().numpy()
                label_np = label.squeeze(dim=1).detach().cpu().numpy()
                empty_label = np.all(label_np == 0, axis=(1, 2))

                for class_idx in range(1, args.classes):
                    hd95_slice = []
                    asd_slice = []
                    for num in range(0, x_out_new.shape[0]):
                        if empty_label[num]:
                            hd95, asd = 0, 0
                        else:
                            pred_binary = pred_np[num] == class_idx
                            label_binary = label_np[num] == class_idx

                            if not label_binary.any():
                                hd95, asd = 0, 0

                            elif not pred_binary.any():
                                hd95, asd = 0, 0
                            else:
                                hd95, asd = calculate_surface_metrics(
                                    pred_binary.astype(np.int32),
                                    label_binary.astype(np.int32),
                                )

                        hd95_slice.append(hd95)
                        asd_slice.append(asd)

                    hd95_class[class_idx - 1][batch_idx] = np.mean(hd95_slice)
                    asd_class[class_idx - 1][batch_idx] = np.mean(asd_slice)

                if is_main_process(args):
                    logger.print(
                        f"Validation epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}:"
                    )
                    logger.print(
                        f"mDice: {mDice}, mIoU: {mIoU}, mHD95: {torch.mean(hd95_class)}, mASD: {torch.mean(asd_class)}"
                    )

    _, IoU, Dice = metric_val.get(mode="")
    HD95 = [torch.mean(hd95_class[index]) for index in range(args.classes - 1)]
    ASD = [torch.mean(asd_class[index]) for index in range(args.classes - 1)]

    if is_main_process(args):
        logger.print(f"Dice: {Dice}, IoU: {IoU}, HD95: {HD95}, ASD: {ASD}")

    return mDice


if __name__ == "__main__":

    args = get_config()
    print(args)
    if args.save == "":
        args.save = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    args.save_path = os.path.join(args.results_dir, args.experiment_name + args.save)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(
        os.path.join(args.runs_dir, args.experiment_name + args.save)
    )

    if args.parallel == "DDP":
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            print(args.device)
    elif args.parallel == "DP":
        args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.checkpoint_finetune_use:
        path_checkpoint = args.path_checkpoint
        print("path_checkpoint: {}".format(path_checkpoint))
        if path_checkpoint == None:
            print("There is no checkpoint file saved !")
        else:
            checkpoint = torch.load(
                path_checkpoint, map_location=lambda storage, loc: storage
            )
            run(writer, args, checkpoint)
    else:

        if args.result_validation:
            print("Visualization! ")
            run(writer, args, None)
        else:
            run(writer, args, None)
