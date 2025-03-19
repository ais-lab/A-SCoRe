import argparse
import os
import random
from distutils.util import strtobool
from pathlib import Path 
from loguru import logger as loguru_logger 

from torch.utils import data
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from lightning.lightning_scr import PL_SCR
from lightning.dataset_pl import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            '--model', nargs='?', type=str, default='scrnet',
            choices=(
                     'sp_sg_attn', 
                     'sp_sg_sparse_separate',
                     'sp_sg_attn_lite')
            )
    parser.add_argument(
            '--device', type=int, default=0
            )
    parser.add_argument(
            '--loss', nargs='?', type=str, default='euclidean_loss',
            choices=('euclidean_loss', 'regcoord_loss')
            )
    parser.add_argument(
            '--dataset', nargs='?', type=str, default='7Scenes',
            choices=('7Scenes', 
                     '7Scenes_SfM', 
                     'Cambridge', 
                     'Cambridge_SfM', 
                     '12Scenes', 
                     '12Scenes_depth')
            )
    parser.add_argument(
            '--scene', nargs='?', type=str, default='chess'
            )
    parser.add_argument(
            '--mode', nargs='?', type=str, default='train',
            choices=('train', 'test')
            )
    parser.add_argument(
            '--pose_mode', nargs='?', type=str, default='hscnet',
            choices=('hscnet', 'dsac', 'poselib', 'pycolmap')
            )
    parser.add_argument(
            '--validate_on_train', nargs='?', type=strtobool, default=False
            )
    parser.add_argument(
            '--pretrained_ckpt', nargs='?', type=str, default=None
            )
    parser.add_argument(
            '--n_iter', nargs='?', type=int, default=300000
            )
    parser.add_argument(
            '--init_lr', nargs='?', type=float, default=1e-4
            )
    parser.add_argument(
            '--batch_size', nargs='?', type=int, default=8
            )
    parser.add_argument(
            '--num_workers', nargs='?', type=int, default=4
            )
    parser.add_argument(
            '--aug', nargs='?', type=strtobool, default=True
            )
    parser.add_argument(
            '--data_path', required=True, type=str
            )
    parser.add_argument(
            '--img_path', required=True, type=str
            )
    parser.add_argument(
            '--log_summary', default='progress_log_summary.txt'
            )
    parser.add_argument(
            '--use_wandb', nargs='?', type=strtobool, default=True
            )
    parser.add_argument(
            '--train_id', nargs='?', type=str, default=''
            )
    parser.add_argument(
            '--dump_dir', type=str, default='checkpoints'
            )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Init seed
    seed = 0
    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
   
    # Init dataset
    dataset_cfg = {'pose': 'gt',
                   'depth': None,
                   'scenecoord': 'pre-computed',
                   'grayscale': True,
                   'resize': (),
                   'augmentation': args.aug}

    dataset = get_dataset(args.dataset)
    if 'SfM' in args.dataset:
        args.data_mode = 'sparse'
        training_dataset = dataset(cfg={},
                                   scene=args.scene,
                                   root=args.data_path,
                                   split='train')
        testing_dataset  = dataset(cfg={},
                                   scene=args.scene,
                                   root=args.data_path,
                                   split='test')
    else:
        args.data_mode = 'dense'
        training_dataset = dataset(cfg=dataset_cfg,
                                   scene=args.scene,
                                   meta_root=args.data_path,
                                   img_root=args.img_path,
                                   split='train')

        testing_dataset = dataset(cfg=dataset_cfg,
                                  scene=args.scene,
                                  meta_root=args.data_path,
                                  img_root=args.img_path,
                                  split='test')

    training_dataloader = data.DataLoader(training_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=args.num_workers,
                                          shuffle=True,
                                          pin_memory=False)

    testing_dataloader = data.DataLoader(testing_dataset,
                                         batch_size=1,
                                         num_workers=0,
                                         shuffle=False,
                                         pin_memory=False)
    
    # Calculate some params for training
    args.len_data = len(training_dataset)
    args.n_epoch = int(np.ceil(args.n_iter * args.batch_size / args.len_data))
    model_id = "{}-{}-{}-initlr{}-iters{}-bsize{}-aug{}-{}".format(\
                args.dataset, args.scene.replace('/','.'),
                args.model, args.init_lr, args.n_iter, args.batch_size, 
                int(args.aug), args.train_id)
    save_path = Path(model_id)
    args.save_path = args.dump_dir / save_path
    args.save_path.mkdir(parents=True, exist_ok=True)
    model = PL_SCR(cfg={},
                   args=args)
    loguru_logger.info("Model initialized")
    
    # Initialize callback for learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]


    if args.mode == 'train':
        args.project = args.train_id
        logger = WandbLogger(project=args.project)
        trainer = pl.Trainer(
                    devices=1,
                    accelerator="gpu",
                    max_epochs=args.n_epoch,
                    callbacks=callbacks,
                    logger=logger
                )
        loguru_logger.info(f"Trainer initialized")
        loguru_logger.info(f"Start training")
        trainer.fit(model, 
                    train_dataloaders=training_dataloader)
    elif args.mode == 'test':
        trainer = pl.Trainer(
                devices=[args.device],
                accelerator="gpu",
                enable_progress_bar = False
                )
        assert args.pretrained_ckpt is not None
        trainer.test(model,
                     dataloaders=testing_dataloader)
        loguru_logger.info(f"Validating on test set done")
        # Changing the batch size to 1
        if args.validate_on_train:
            loguru_logger.info(f"Validating on the train set")
            training_dataloader_cp = data.DataLoader(training_dataset,
                                                   batch_size=1,
                                                   num_workers=0,
                                                   shuffle=False)
            trainer.test(model,
                         dataloaders=training_dataloader_cp)
