import argparse
import os
import shutil
import sys
import time
import logging
from collections import namedtuple
from itertools import repeat

import yaml
from tensorboardX import SummaryWriter

from nets import Model
from dataset import CREStereoDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler


def parse_yaml(file_path: str) -> namedtuple:
    """Parse yaml configuration file and return the object in `namedtuple`."""
    with open(file_path, "rb") as f:
        cfg: dict = yaml.safe_load(f)
    args = namedtuple("train_args", cfg.keys())(*cfg.values())
    # save cfg into train_log
    ensure_dir(args.log_dir)
    dst_file = os.path.join(args.log_dir, file_path.split('/')[-1])
    shutil.copy2(file_path, dst_file)    
    return args


def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return "{:02d}:{:02d}:{:02d}".format(hour, minute, seconds)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def adjust_learning_rate(optimizer, epoch):

    warm_up = 0.02
    const_range = 0.6
    min_lr_rate = 0.05

    if epoch <= args.n_total_epoch * warm_up:
        lr = (1 - min_lr_rate) * args.base_lr / (
            args.n_total_epoch * warm_up
        ) * epoch + min_lr_rate * args.base_lr
    elif args.n_total_epoch * warm_up < epoch <= args.n_total_epoch * const_range:
        lr = args.base_lr
    else:
        lr = (min_lr_rate - 1) * args.base_lr / (
            (1 - const_range) * args.n_total_epoch
        ) * epoch + (1 - min_lr_rate * const_range) / (1 - const_range) * args.base_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8):
    '''
    valid: (2, 384, 512) (B, H, W) -> (B, 1, H, W)
    flow_preds[0]: (B, 2, H, W)
    flow_gt: (B, 2, H, W)
    '''
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = torch.abs(flow_preds[i] - flow_gt)
        flow_loss += i_weight * (valid.unsqueeze(1) * i_loss).mean()

    return flow_loss

def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

def train_dist(args, world_size):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",type=int)
    FLAGS = parser.parse_args()
    local_rank = FLAGS.local_rank
    # directory check
    log_model_dir = os.path.join(args.log_dir, "models")
    ensure_dir(log_model_dir)

    # distributed init and model / optimizer
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl is highly recommanded
    model = Model(
        max_disp=args.max_disp, mixed_precision=args.mixed_precision, test_mode=False
    )
    # sync batch norm
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

    if dist.get_rank() == 0:
        # tensorboard
        tb_log = SummaryWriter(os.path.join(args.log_dir, "train.events"))

        # worklog
        logging.basicConfig(level=eval(args.log_level))
        worklog = logging.getLogger("train_logger")
        worklog.propagate = False
        fileHandler = logging.FileHandler(
            os.path.join(args.log_dir, "worklog.txt"), mode="a", encoding="utf8"
        )
        formatter = logging.Formatter(
            fmt="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
        )
        fileHandler.setFormatter(formatter)
        consoleHandler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="\x1b[32m%(asctime)s\x1b[0m %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
        )
        consoleHandler.setFormatter(formatter)
        worklog.handlers = [fileHandler, consoleHandler]

        # params stat
        worklog.info(f"Use {world_size} GPU(s)")
        worklog.info("Params: %s" % sum([p.numel() for p in model.parameters()]))

    # load pretrained model if exist
    chk_path = os.path.join(log_model_dir, "latest.pth")
    if args.loadmodel is not None:
        chk_path = args.loadmodel
    elif not os.path.exists(chk_path):
        chk_path = None

    if chk_path is not None:
        if dist.get_rank() == 0:
            worklog.info(f"loading model: {chk_path}")
        # map_location=torch.device('cpu') make more balance memory usage
        state_dict = torch.load(chk_path, map_location=torch.device('cpu'))
        model.module.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optim_state_dict'])
        resume_epoch_idx = state_dict["epoch"]
        resume_iters = state_dict["iters"]
        start_epoch_idx = resume_epoch_idx + 1
        start_iters = resume_iters
    else:
        start_epoch_idx = 1
        start_iters = 0

    # datasets
    dataset = CREStereoDataset(args.training_data_path)
    # dataset = MixDataset("train", 
    #     data_path=args.data["train"]["data_path"],
    #     fields=args.data["train"]["fields"],
    #     filelists=args.data["train"]["filelists"],
    #     input_size=(args.data["train"]["input_size"][0], args.data["train"]["input_size"][1]))
    if dist.get_rank() == 0:
        worklog.info(f"Dataset size: {len(dataset)}")
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, 
        batch_size=args.batch_size, num_workers=4, sampler=train_sampler)

    # counter
    cur_iters = start_iters
    total_iters = args.minibatch_per_epoch * args.n_total_epoch
    t0 = time.perf_counter()
    for epoch_idx in range(start_epoch_idx, args.n_total_epoch + 1):
        dataloader.sampler.set_epoch(epoch_idx)
        # adjust learning rate
        epoch_total_train_loss = 0     
        adjust_learning_rate(optimizer, epoch_idx)
        model.train()

        t1 = time.perf_counter()

        for batch_idx, mini_batch_data in enumerate(dataloader):

            if batch_idx % args.minibatch_per_epoch == 0 and batch_idx != 0:
                break
            cur_iters += 1

            # parse data
            left, right, gt_disp, valid_mask = (
                mini_batch_data["left"].to(local_rank),
                mini_batch_data["right"].to(local_rank),
                mini_batch_data["disparity"].to(local_rank),
                mini_batch_data["mask"].to(local_rank),
            )

            t2 = time.perf_counter()
            optimizer.zero_grad()

            # pre-process
            gt_disp = torch.unsqueeze(gt_disp, dim=1)  # [2, 384, 512] -> [2, 1, 384, 512]
            gt_flow = torch.cat([gt_disp, gt_disp * 0], dim=1)  # [2, 2, 384, 512]

            # forward
            flow_predictions = model(left, right)

            # loss & backword
            loss = sequence_loss(
                flow_predictions, gt_flow, valid_mask, gamma=0.8
            ).to(local_rank)

            # loss stats
            loss_item = loss.data.item()
            epoch_total_train_loss += loss_item
            loss.backward()
            optimizer.step()
            t3 = time.perf_counter()
            if dist.get_rank() == 0:
                if cur_iters % 10 == 0:
                    tdata = t2 - t1
                    time_train_passed = t3 - t0
                    time_iter_passed = t3 - t1
                    step_passed = cur_iters - start_iters
                    eta = (
                        (total_iters - cur_iters)
                        / max(step_passed, 1e-7)
                        * time_train_passed
                    )

                    meta_info = list()
                    meta_info.append("{:.2g} b/s".format(1.0 / time_iter_passed))
                    meta_info.append("passed:{}".format(format_time(time_train_passed)))
                    meta_info.append("eta:{}".format(format_time(eta)))
                    meta_info.append(
                        "data_time:{:.2g}".format(tdata / time_iter_passed)
                    )

                    meta_info.append(
                        "lr:{:.5g}".format(optimizer.param_groups[0]["lr"])
                    )
                    meta_info.append(
                        "[{}/{}:{}/{}]".format(
                            epoch_idx,
                            args.n_total_epoch,
                            batch_idx,
                            args.minibatch_per_epoch,
                        )
                    )
                    loss_info = list()
                    loss_info.append("{}:{:.4g}".format("total_loss", loss_item))
                    # exp_name = ['\n' + os.path.basename(os.getcwd())]

                    info = [",".join(meta_info+loss_info)]
                    worklog.info("".join(info))

                    # minibatch loss
                    tb_log.add_scalar("train/loss_batch", loss_item, cur_iters)
                    tb_log.add_scalar(
                        "train/lr", optimizer.param_groups[0]["lr"], cur_iters
                    )
                    tb_log.flush()

            t1 = time.perf_counter()

        if dist.get_rank() == 0:
            # epoch loss
            tb_log.add_scalar(
                "train/loss",
                epoch_total_train_loss / args.minibatch_per_epoch,
                epoch_idx,
            )
            tb_log.flush()

            # save model params
            ckp_data = {
                "epoch": epoch_idx,
                "iters": cur_iters,
                "batch_size": args.batch_size * world_size,
                "epoch_size": args.minibatch_per_epoch,
                "train_loss": epoch_total_train_loss / args.minibatch_per_epoch,
                "state_dict": model.module.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
            }
            torch.save(ckp_data, os.path.join(log_model_dir, "latest.pth"))
            if epoch_idx % args.model_save_freq_epoch == 0:
                save_path = os.path.join(log_model_dir, "epoch-%d.pth" % epoch_idx)
                worklog.info(f"Model params saved: {save_path}")
                torch.save(ckp_data, save_path)
    if dist.get_rank() == 0:
        worklog.info("Training is done, exit.")

def train(args, world_size):
    # directory check
    log_model_dir = os.path.join(args.log_dir, "models")
    ensure_dir(log_model_dir)

    # model / optimizer
    model = Model(
        max_disp=args.max_disp, mixed_precision=args.mixed_precision, test_mode=False
    )
    model = nn.DataParallel(model,device_ids=[i for i in range(world_size)])
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

    tb_log = SummaryWriter(os.path.join(args.log_dir, "train.events"))

    # worklog
    logging.basicConfig(level=eval(args.log_level))
    worklog = logging.getLogger("train_logger")
    worklog.propagate = False
    fileHandler = logging.FileHandler(
        os.path.join(args.log_dir, "worklog.txt"), mode="a", encoding="utf8"
    )
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    fileHandler.setFormatter(formatter)
    consoleHandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="\x1b[32m%(asctime)s\x1b[0m %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    consoleHandler.setFormatter(formatter)
    worklog.handlers = [fileHandler, consoleHandler]

    # params stat
    worklog.info(f"Use {world_size} GPU(s)")
    worklog.info("Params: %s" % sum([p.numel() for p in model.parameters()]))

    # load pretrained model if exist
    chk_path = os.path.join(log_model_dir, "latest.pth")
    if args.loadmodel is not None:
        chk_path = args.loadmodel
    elif not os.path.exists(chk_path):
        chk_path = None

    if chk_path is not None:
        worklog.info(f"loading model: {chk_path}")
        state_dict = torch.load(chk_path)
        model.module.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optim_state_dict'])
        resume_epoch_idx = state_dict["epoch"]
        resume_iters = state_dict["iters"]
        start_epoch_idx = resume_epoch_idx + 1
        start_iters = resume_iters
    else:
        start_epoch_idx = 1
        start_iters = 0

    # datasets
    dataset = CREStereoDataset(args.training_data_path)
    sampler = RandomSampler(dataset, replacement=False)
    worklog.info(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size*world_size,
                            num_workers=0, drop_last=True, persistent_workers=False, pin_memory=True)
    dataloader = repeater(dataloader)

    # counter
    cur_iters = start_iters
    total_iters = args.minibatch_per_epoch * args.n_total_epoch
    t0 = time.perf_counter()
    for epoch_idx in range(start_epoch_idx, args.n_total_epoch + 1):

        # adjust learning rate
        epoch_total_train_loss = 0
        adjust_learning_rate(optimizer, epoch_idx)
        model.train()

        t1 = time.perf_counter()

        # for mini_batch_data in dataloader:
        for batch_idx, mini_batch_data in enumerate(dataloader):

            if batch_idx % args.minibatch_per_epoch == 0 and batch_idx != 0:
                break
            cur_iters += 1

            # parse data
            left, right, gt_disp, valid_mask = (
                mini_batch_data["left"].cuda(),
                mini_batch_data["right"].cuda(),
                mini_batch_data["disparity"].cuda(),
                mini_batch_data["mask"].cuda(),
            )

            t2 = time.perf_counter()
            optimizer.zero_grad()

            # pre-process
            gt_disp = torch.unsqueeze(gt_disp, dim=1)  # [2, 384, 512] -> [2, 1, 384, 512]
            gt_flow = torch.cat([gt_disp, gt_disp * 0], dim=1)  # [2, 2, 384, 512]

            # forward
            flow_predictions = model(left, right)

            # loss & backword
            loss = sequence_loss(
                flow_predictions, gt_flow, valid_mask, gamma=0.8
            )

            # loss stats
            loss_item = loss.data.item()
            epoch_total_train_loss += loss_item
            loss.backward()
            optimizer.step()
            t3 = time.perf_counter()

            if cur_iters % 10 == 0:
                tdata = t2 - t1
                time_train_passed = t3 - t0
                time_iter_passed = t3 - t1
                step_passed = cur_iters - start_iters
                eta = (
                    (total_iters - cur_iters)
                    / max(step_passed, 1e-7)
                    * time_train_passed
                )

                meta_info = list()
                meta_info.append("{:.2g} b/s".format(1.0 / time_iter_passed))
                meta_info.append("passed:{}".format(format_time(time_train_passed)))
                meta_info.append("eta:{}".format(format_time(eta)))
                meta_info.append(
                    "data_time:{:.2g}".format(tdata / time_iter_passed)
                )

                meta_info.append(
                    "lr:{:.5g}".format(optimizer.param_groups[0]["lr"])
                )
                meta_info.append(
                    "[{}/{}:{}/{}]".format(
                        epoch_idx,
                        args.n_total_epoch,
                        batch_idx,
                        args.minibatch_per_epoch,
                    )
                )
                loss_info = [" ==> {}:{:.4g}".format("loss", loss_item)]
                # exp_name = ['\n' + os.path.basename(os.getcwd())]

                info = [",".join(meta_info)] + loss_info
                worklog.info("".join(info))

                # minibatch loss
                tb_log.add_scalar("train/loss_batch", loss_item, cur_iters)
                tb_log.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], cur_iters
                )
                tb_log.flush()

            t1 = time.perf_counter()

        tb_log.add_scalar(
            "train/loss",
            epoch_total_train_loss / args.minibatch_per_epoch,
            epoch_idx,
        )
        tb_log.flush()

        # save model params
        ckp_data = {
            "epoch": epoch_idx,
            "iters": cur_iters,
            "batch_size": args.batch_size*world_size,
            "epoch_size": args.minibatch_per_epoch,
            "train_loss": epoch_total_train_loss / args.minibatch_per_epoch,
            "state_dict": model.module.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        }
        torch.save(ckp_data, os.path.join(log_model_dir, "latest.pth"))
        if epoch_idx % args.model_save_freq_epoch == 0:
            save_path = os.path.join(log_model_dir, "epoch-%d.pth" % epoch_idx)
            worklog.info(f"Model params saved: {save_path}")
            torch.save(ckp_data, save_path)

    worklog.info("Training is done, exit.")

def main(args):
    # initial info
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    world_size = torch.cuda.device_count()  # number of GPU(s)
    cudnn.benchmark = True
    if args.dist and world_size > 1:
        train_dist(args, world_size)
    else:
        train(args, world_size)

if __name__ == "__main__":
    # train configuration
    args = parse_yaml("cfgs/train.yaml")
    main(args)
