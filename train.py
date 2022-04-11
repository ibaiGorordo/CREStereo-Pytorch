import os
import sys
import time
import logging
from collections import namedtuple

import yaml
from tensorboardX import SummaryWriter

from nets import Model
from dataset import CREStereoDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def parse_yaml(file_path: str) -> namedtuple:
    """Parse yaml configuration file and return the object in `namedtuple`."""
    with open(file_path, "rb") as f:
        cfg: dict = yaml.safe_load(f)
    args = namedtuple("train_args", cfg.keys())(*cfg.values())
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


def main(args):
    # initial info
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # rank, world_size = dist.get_rank(), dist.get_world_size()
    world_size = torch.cuda.device_count()  # number of GPU(s)

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
    # model = nn.DataParallel(model,device_ids=[0])

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
        # if rank == 0:
        worklog.info(f"loading model: {chk_path}")
        state_dict = torch.load(chk_path)
        model.load_state_dict(state_dict['state_dict'])
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
    # if rank == 0:
    worklog.info(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True, persistent_workers=False, pin_memory=True)

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
        # batch_idx = 0

        # for mini_batch_data in dataloader:
        for batch_idx, mini_batch_data in enumerate(dataloader):

            if batch_idx % args.minibatch_per_epoch == 0 and batch_idx != 0:
                break
            # batch_idx += 1
            cur_iters += 1

            # parse data
            left, right, gt_disp, valid_mask = (
                mini_batch_data["left"],
                mini_batch_data["right"],
                mini_batch_data["disparity"].cuda(),
                mini_batch_data["mask"].cuda(),
            )

            t2 = time.perf_counter()
            optimizer.zero_grad()

            # pre-process
            gt_disp = torch.unsqueeze(gt_disp, dim=1)  # [2, 384, 512] -> [2, 1, 384, 512]
            gt_flow = torch.cat([gt_disp, gt_disp * 0], dim=1)  # [2, 2, 384, 512]

            # forward
            flow_predictions = model(left.cuda(), right.cuda())

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
            "batch_size": args.batch_size,
            "epoch_size": args.minibatch_per_epoch,
            "train_loss": epoch_total_train_loss / args.minibatch_per_epoch,
            "state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        }
        torch.save(ckp_data, os.path.join(log_model_dir, "latest.pth"))
        if epoch_idx % args.model_save_freq_epoch == 0:
            save_path = os.path.join(log_model_dir, "epoch-%d.pth" % epoch_idx)
            worklog.info(f"Model params saved: {save_path}")
            torch.save(ckp_data, save_path)

    worklog.info("Training is done, exit.")


if __name__ == "__main__":
    # train configuration
    args = parse_yaml("cfgs/train.yaml")
    main(args)
