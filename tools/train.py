import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict, tiny_imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import dump_cfg
from mdistiller.engine import trainer_dict


def main(cfg, resume, opts):
    IS_MASTER = bool(int(os.environ['IS_MASTER_NODE']))
    
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if IS_MASTER:
        if cfg.LOG.WANDB:
            try:
                import wandb

                wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
            except:
                print(log_msg("Failed to use WANDB", "INFO"))
                cfg.LOG.WANDB = False

    # cfg & loggers
    if IS_MASTER:
        dump_cfg(cfg, show=True)
    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        elif cfg.DATASET.TYPE == "tiny_imagenet":
            model_student = tiny_imagenet_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        if IS_MASTER:
            print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_dict = tiny_imagenet_model_dict if cfg.DATASET.TYPE == "tiny_imagenet" else cifar_model_dict
            net, pretrain_model_path = model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    distiller = distiller.cuda()
    distiller = DistributedDataParallel(distiller, device_ids=[rank], find_unused_parameters=True)

    if (cfg.DISTILLER.TYPE != "NONE") and IS_MASTER:
        print(
            log_msg(
                "Extra parameters of {}: {:,d}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse
    
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    os.environ['IS_MASTER_NODE'] = str(int(rank == 0))

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default=[], nargs='*')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    for cfg_fn in args.cfg:
        cfg.merge_from_file(cfg_fn)
    cfg.merge_from_list(args.opts)
    cfg.EXPERIMENT.DDP = True
    cfg.DATASET.NUM_WORKERS //= world_size
    cfg.DATASET.TEST.BATCH_SIZE //= world_size
    cfg.SOLVER.BATCH_SIZE //= world_size
    cfg.freeze()
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    try:
        main(cfg, args.resume, args.opts)
    except KeyboardInterrupt:
        pass
    finally:
        dist.destroy_process_group()
