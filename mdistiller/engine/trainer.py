import os
from glob import glob
import shutil
import contextlib
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.amp as amp
from collections import OrderedDict
import getpass
from tensorboardX import SummaryWriter
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
)
from .dot import DistillationOrientedTrainer
from ..utils import dist_fn
from ..engine.cfg import dump_cfg


def update_loss_meters(
    loss_meters: dict[str, AverageMeter],
    losses_dict: dict[str, torch.Tensor],
    batch_size: int,
):
    for k, v in losses_dict.items():
        if k not in loss_meters:
            loss_meters[k] = AverageMeter()
        loss_meters[k].update(v, batch_size)
    return loss_meters


class BaseTrainer(object):
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        IS_MASTER = bool(int(os.environ['IS_MASTER_NODE']))
        
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self.init_optimizer(cfg)
        self.best_acc = -1

        username = getpass.getuser()
        # init loggers
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if IS_MASTER:
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))
        else:
            self.tf_writer = None
        
        # dump config and distiller code
        code_path = os.path.join(self.log_path, 'code')
        os.makedirs(code_path, exist_ok=True)
        with open(os.path.join(code_path, '_cfg.yaml'), 'w') as file:
            cfg_dump = dump_cfg(cfg, show=False)
            print(cfg_dump, end='', file=file)
        
        distiller_names = glob(f'./mdistiller/distillers/**/*.py', recursive=True)
        target_name = './mdistiller/distillers/' + cfg.DISTILLER.TYPE.replace('.', os.sep).lower() + '.py'
        distiller_name = None
        for fname in distiller_names:
            if target_name == fname.lower():
                distiller_name = fname
                break
        shutil.copyfile(distiller_name, os.path.join(code_path, f'distiller.py'))
        
        self.use_amp = cfg.EXPERIMENT.AMP
        self.amp_scaler = amp.GradScaler('cuda') if self.use_amp else None
        self.grad_clip = cfg.SOLVER.GRAD_CLIP

    def init_optimizer(self, cfg):
        match cfg.SOLVER.TYPE:
            case 'SGD':
                optimizer = optim.SGD(
                    self.distiller.module.get_learnable_parameters(),
                    lr=cfg.SOLVER.LR,
                    momentum=cfg.SOLVER.SGD.MOMENTUM,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                )
            case 'Adam':
                optimizer = optim.Adam(
                    self.distiller.module.get_learnable_parameters(),
                    lr=cfg.SOLVER.LR,
                    betas=cfg.SOLVER.ADAM.BETAS,
                    eps=cfg.SOLVER.ADAM.EPSILON,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                )
            case 'AdamW':
                optimizer = optim.AdamW(
                    self.distiller.module.get_learnable_parameters(),
                    lr=cfg.SOLVER.LR,
                    betas=cfg.SOLVER.ADAM.BETAS,
                    eps=cfg.SOLVER.ADAM.EPSILON,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                )
            case _:
                raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def log(self, lr, epoch, log_dict):
        # tensorboard log
        for k, v in log_dict.items():
            if isinstance(v, dict):
                for name, value in v.items():
                    self.tf_writer.add_scalar(f'{k}/{name}', value, epoch)
            else:
                self.tf_writer.add_scalar(k, v, epoch)
        self.tf_writer.flush()
        # wandb log
        if self.cfg.LOG.WANDB:
            import wandb

            wandb.log({"current lr": lr})
            wandb.log(log_dict)
        if log_dict["test_acc"] > self.best_acc:
            self.best_acc = log_dict["test_acc"]
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_acc"] = self.best_acc
        # worklog.txt
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            lines = [
                "-" * 35 + os.linesep,
                "epoch: {}".format(epoch) + os.linesep,
                "lr: {:.4f}".format(float(lr)) + os.linesep,
            ]
            for k, v in log_dict.items():
                match v:
                    case int():
                        lines.append("{}: {:d}".format(k, v) + os.linesep)
                    case float() | np.ndarray() | torch.Tensor():
                        lines.append("{}: {:.4f}".format(k, v) + os.linesep)
                    case dict():
                        lines.append('{}:'.format(k) + os.linesep)
                        for name, value in v.items():
                            lines.append("    {}: {:.4f}".format(name, value) + os.linesep)
            lines.append("-" * 35 + os.linesep)
            writer.writelines(lines)
        # worklog.yaml
        with open(os.path.join(self.log_path, 'worklog.yaml'), 'a') as writer:
            lines = [
                f'- epoch: {epoch}{os.linesep}',
                f'  lr: {float(lr):.4f}{os.linesep}',
            ]
            for k, v in log_dict.items():
                match v:
                    case int():
                        lines.append(f"  {k}: {v:d}{os.linesep}")
                    case float() | np.ndarray() | torch.Tensor():
                        lines.append(f"  {k}: {v:.4f}{os.linesep}")
                    case dict():
                        lines.append(f'  {k}:{os.linesep}')
                        for name, value in v.items():
                            lines.append(f"    {name}: {value:.4f}{os.linesep}")
            lines.append('\n')
            writer.writelines(lines)

    def train(self, resume=False):
        IS_MASTER = bool(int(os.environ['IS_MASTER_NODE']))
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        if IS_MASTER:
            print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
            with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
                writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_epoch(self, epoch):
        IS_MASTER = bool(int(os.environ['IS_MASTER_NODE']))
        
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": dict(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar = range(num_iter)
        if IS_MASTER:
            pbar = tqdm(pbar, dynamic_ncols=True)

        # train loops
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            lr = adjust_learning_rate(epoch, idx, self.cfg, self.optimizer)
            msg = self.train_iter(data, epoch, train_meters)
            if IS_MASTER:
                pbar.set_description(log_msg(msg, "TRAIN"))
                pbar.update()
        if IS_MASTER:
            pbar.close()

        # validate
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller)

        # log
        if IS_MASTER:
            log_dict = OrderedDict(
                {
                    "train_acc": train_meters["top1"].avg,
                    "train_loss": {
                        k.replace('loss_', ''): v.avg
                        for k, v in train_meters["losses"].items()
                    },
                    "test_acc": test_acc,
                    "test_acc_top5": test_acc_top5,
                    "test_loss": test_loss,
                }
            )
            self.log(lr, epoch, log_dict)
            # saving checkpoint
            state = {
                "epoch": epoch,
                "model": self.distiller.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_acc": self.best_acc,
            }
            student_state = {"model": self.distiller.module.student.state_dict()}
            save_checkpoint(state, os.path.join(self.log_path, "latest"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_latest")
            )
            if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
                save_checkpoint(
                    state, os.path.join(self.log_path, "epoch_{}".format(epoch))
                )
                save_checkpoint(
                    student_state,
                    os.path.join(self.log_path, "student_{}".format(epoch)),
                )
            # update the best
            if test_acc >= self.best_acc:
                save_checkpoint(state, os.path.join(self.log_path, "best"))
                save_checkpoint(
                    student_state, os.path.join(self.log_path, "student_best")
                )

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        with amp.autocast('cuda') if self.use_amp else contextlib.nullcontext():
            preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)
            loss: torch.Tensor = sum(losses_dict.values())

        # backward
        self.backward_loss(loss)
        train_meters["training_time"].update(time.time() - train_start_time)
            
        # collect info
        preds_all = dist_fn.gather(preds)
        target_all = dist_fn.gather(target)
        loss = dist_fn.reduce(loss, dist.ReduceOp.AVG)
        losses_dict = {
            k: dist_fn.reduce(v, dist.ReduceOp.AVG).cpu().detach().numpy().mean()
            for k, v in losses_dict.items()
        }
        losses_dict['_total'] = loss.cpu().detach().numpy().mean()
            
        batch_size = len(preds_all)
        acc1, acc5 = accuracy(preds_all, target_all, topk=(1, 5))
        update_loss_meters(train_meters["losses"], losses_dict, batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
            
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"]['_total'].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg
    
    def backward_loss(self, loss: torch.Tensor):
        if self.use_amp:
            self.amp_scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.amp_scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.distiller.module.get_learnable_parameters(), 
                    self.grad_clip
                )
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.distiller.module.get_learnable_parameters(), 
                    self.grad_clip
                )
            self.optimizer.step()


class CRDTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        with amp.autocast('cuda') if self.use_amp else contextlib.nullcontext():
            preds, losses_dict = self.distiller(
                image=image, target=target, index=index, contrastive_index=contrastive_index
            )
            loss: torch.Tensor = sum(losses_dict.values())

        # backward
        self.backward_loss(loss)
        train_meters["training_time"].update(time.time() - train_start_time)
        
        # collect info
        preds_all = dist_fn.gather(preds)
        target_all = dist_fn.gather(target)
        loss = dist_fn.reduce(loss, dist.ReduceOp.AVG)
        losses_dict = {
            k: dist_fn.reduce(v, dist.ReduceOp.AVG).cpu().detach().numpy().mean()
            for k, v in losses_dict.items()
        }
        losses_dict['_total'] = loss.cpu().detach().numpy().mean()
        
        batch_size = len(preds_all)
        acc1, acc5 = accuracy(preds_all, target_all, topk=(1, 5))
        update_loss_meters(train_meters["losses"], losses_dict, batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"]['_total'].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class DOT(BaseTrainer):
    def init_optimizer(self, cfg):
        if self.use_amp:
            raise NotImplementedError('AMP is not supported for DOT.') 
        if cfg.SOLVER.TYPE == "SGD":
            m_task = cfg.SOLVER.MOMENTUM - cfg.SOLVER.DOT.DELTA
            m_kd = cfg.SOLVER.MOMENTUM + cfg.SOLVER.DOT.DELTA
            optimizer = DistillationOrientedTrainer(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=m_task,
                momentum_kd=m_kd,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, resume=False):
        IS_MASTER = bool(int(os.environ['IS_MASTER_NODE']))
        
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        
        if IS_MASTER:
            print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
            with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
                writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_iter(self, data, epoch, train_meters):
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)

        # dot backward
        loss_ce: torch.Tensor = losses_dict['loss_ce']
        loss_kd: torch.Tensor = losses_dict['loss_kd']
        self.optimizer.zero_grad(set_to_none=True)
        loss_kd.backward(retain_graph=True)
        self.optimizer.step_kd()
        self.optimizer.zero_grad(set_to_none=True)
        loss_ce.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)

        # collect info
        preds_all = dist_fn.gather(preds)
        target_all = dist_fn.gather(target)
        loss_ce = dist_fn.reduce(loss_ce, dist.ReduceOp.AVG)
        loss_kd = dist_fn.reduce(loss_kd, dist.ReduceOp.AVG)
        losses_dict = {
            k: dist_fn.reduce(v, dist.ReduceOp.AVG).cpu().detach().numpy().mean()
            for k, v in losses_dict.items()
        }
        losses_dict['_total'] = (loss_kd + loss_ce).cpu().detach().numpy().mean()
        
        batch_size = len(preds_all)
        acc1, acc5 = accuracy(preds_all, target_all, topk=(1, 5))
        update_loss_meters(train_meters["losses"], losses_dict, batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"]['_total'].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CRDDOT(BaseTrainer):

    def init_optimizer(self, cfg):
        if self.use_amp:
            raise NotImplementedError('AMP is not supported for DOT.') 
        if cfg.SOLVER.TYPE == "SGD":
            m_task = cfg.SOLVER.MOMENTUM - cfg.SOLVER.DOT.DELTA
            m_kd = cfg.SOLVER.MOMENTUM + cfg.SOLVER.DOT.DELTA
            optimizer = DistillationOrientedTrainer(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=m_task,
                momentum_kd=m_kd,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, resume=False):
        IS_MASTER = bool(int(os.environ['IS_MASTER_NODE']))
        
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
            
        if IS_MASTER:
            print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
            with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
                writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # dot backward
        loss_ce, loss_kd = losses_dict['loss_ce'].mean(), losses_dict['loss_kd'].mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_kd.backward(retain_graph=True)
        self.optimizer.step_kd()
        self.optimizer.zero_grad(set_to_none=True)
        loss_ce.backward()
        # self.optimizer.step((1 - epoch / 240.))
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)

        # collect info
        preds_all = dist_fn.gather(preds)
        target_all = dist_fn.gather(target)
        loss_ce = dist_fn.reduce(loss_ce, dist.ReduceOp.AVG)
        loss_kd = dist_fn.reduce(loss_kd, dist.ReduceOp.AVG)
        losses_dict = {
            k: dist_fn.reduce(v, dist.ReduceOp.AVG).cpu().detach().numpy().mean()
            for k, v in losses_dict.items()
        }
        losses_dict['_total'] = (loss_ce + loss_kd).cpu().detach().numpy().mean()
        
        batch_size = len(preds_all)
        acc1, acc5 = accuracy(preds_all, target_all, topk=(1, 5))
        update_loss_meters(train_meters["losses"], losses_dict, batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"]['_total'].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg
