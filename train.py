import argparse
import logging
import os
import time
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed

from config import get_config
from datasets.dataset import MXFaceDataset
from datasets.common import DataLoaderX
from utils.utils_amp import MaxClipGradScaler
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory
from loss.loss_def import LossFactory

class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.

    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """

    def __init__(self, backbone_factory, head_factory):
        """Init face model by backbone factorcy and head factory.

        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone_factory.get_backbone()
        self.head = head_factory.get_head()

    def forward(self, data):
        feat = self.backbone.forward(data)
        pred = self.head.forward(feat)
        return pred

def main(args):
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    except KeyError:
        world_size = 1
        rank = 0
        dist_url = "tcp://127.0.0.1:12583"


    dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    cfg = get_config(args.dataset)
    if args.output:
        cfg.output = args.output
    if args.batchsize:
        cfg.batch_size = args.batchsize

    if not os.path.exists(cfg.output) and rank is 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)
    train_set = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)

    num_class = cfg.num_classes
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)

    backbone_factory = BackboneFactory(
        args.backbone_type, args.backbone_conf_file)
    head_factory = HeadFactory(args.head_type, args.head_conf_file, num_class=num_class)
    # backbone = backbone_factory.get_backbone().to(local_rank)
    # head = head_factory.get_head().to(local_rank)

    model = FaceModel(backbone_factory, head_factory).to(local_rank)
    loss_factory = LossFactory(args.loss_type, args.loss_conf_file, local_rank)
    criterion = loss_factory.get_loss().cuda()
    opt = torch.optim.SGD(
        params=[{'params': model.backbone.parameters(), 'lr': cfg.lr},
                {'params': model.head.parameters(), 'lr': cfg.lr},
                {'params': criterion.parameters(), 'lr': 0.02 * cfg.lr},
                ],
        momentum=0.9, weight_decay=cfg.weight_decay)

    if args.resume_backbone:
        checkpoint = torch.load(args.resume_backbone)
        model.backbone.load_state_dict(checkpoint)
        print("Resume: backbone resume successfully from {}!".format(args.resume_backbone))

    if args.resume_head:
        checkpoint = torch.load(args.resume_head)
        model.backbone.load_state_dict(checkpoint)
        print("Resume: head resume successfully from {}!".format(args.resume_head))

    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[local_rank], find_unused_parameters=True)
    model.train()

    scheduler_lr = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt, lr_lambda=cfg.lr_func)

    start_epoch = args.start_epoch
    total_step = int(len(train_set) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank is 0: logging.info("Total Step is: %d" % total_step)

    callback_verification = CallBackVerification(args.eval_steps, rank, cfg.val_targets, cfg.val_root, flip_test=args.flip_test)
    callback_logging = CallBackLogging(args.log_steps, rank, total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = 0
    grad_amp = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None

    for epoch in range(start_epoch, cfg.num_epoch):
        scheduler_lr.step(epoch=epoch)
        train_sampler.set_epoch(epoch)
        for step, (img, labels) in enumerate(train_loader):
            global_step += 1
            preds = model(img)
            if args.loss_type == 'AdaABLoss':
                if step < 100:
                    fast_update = True
                else:
                    fast_update = False
                loss_args = {"epoch": epoch, "fast_update": fast_update}
                loss_batch = criterion(preds, labels, **loss_args)
            else:
                loss_batch = criterion(preds, labels)
            if not isinstance(loss_batch, dict):
                loss_val = loss_batch
            else:
                loss_val = 0
                for key, item in loss_batch.items():
                    if 'loss' in key:
                        loss_val += item
            opt.zero_grad()
            loss_val.backward()
            opt.step()
            loss.update(loss_val, 1)
            callback_logging(global_step, loss, epoch, cfg.fp16, grad_amp, opt)
            callback_verification(global_step, model.module.backbone)
            if global_step % args.eval_steps == 0:
                callback_checkpoint(global_step, model.module.backbone, model.module.head, epoch=epoch+1)
            if step % args.log_steps == 0 and local_rank == 0:
                if isinstance(loss_batch, dict):
                    line = ""
                    for key, item in loss_batch.items():
                        if key == 'loss_cov':
                            line += "{}: {}   ".format("coverage", 0.99 - math.sqrt(item / 32))
                        else:
                            line += "{}: {}   ".format(key, item)
                    print(line)

        callback_checkpoint(None, model.module.backbone, model.module.head, epoch=epoch+1)
        
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--resume_backbone', type=str, default='', help='backbone resuming')
    parser.add_argument('--resume_head', type=str, default='', help='head resuming')
    parser.add_argument("--dataset", default='ms1m', type=str,
                      help="dataset")
    parser.add_argument("--backbone_type", type=str,
                      help="Mobilefacenets, Resnet.")
    parser.add_argument("--backbone_conf_file", type=str, default='./configs/backbone_conf.yaml',
                      help="the path of backbone_conf.yaml.")
    parser.add_argument("--head_type", type=str,
                      help="mv-softmax, arcface, npc-face.")
    parser.add_argument("--head_conf_file", type=str, default='./configs/head_conf.yaml',
                      help="the path of head_conf.yaml.")
    parser.add_argument("--loss_type", type=str,
                      help="")
    parser.add_argument("--loss_conf_file", type=str, default='./configs/loss_conf.yaml',
                      help="the path of backbone_conf.yaml.")
    parser.add_argument("--log_steps", type=int, default=10,
                      help="log interval steps")
    parser.add_argument("--start_epoch", type=int, default=0,
                      help="start epoch")
    parser.add_argument("--eval_steps", type=int, default=2000,
                      help="log interval steps")
    parser.add_argument("--output", type=str, default='',
                      help="Output dir")
    parser.add_argument("--batchsize", type=int, default=64,
                      help="batch size")
    parser.add_argument("--flip_test", action="store_true")
    args_ = parser.parse_args()
    main(args_)
