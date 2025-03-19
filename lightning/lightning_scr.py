import sys
import os
import torch
import time
import cv2
import numpy as np
import poselib
import pytorch_lightning as pl
from loguru import logger as loguru_logger

from .models import get_model
from .loss import get_loss
from .utils import adjust_lr, save_state
from .dataset_pl.utils import load_intrinsic_sfm, qvec2rotmat

sys.path.insert(0, '../pnpransac')
from pnpransac import pnpransac

def get_pose_err(pose_gt, pose_est):
    transl_err = np.linalg.norm(pose_gt[0:3,3]-pose_est[0:3,3])
    rot_err = pose_est[0:3,0:3].T.dot(pose_gt[0:3,0:3])
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1,3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis = 1), -1) / np.pi * 180.
    return transl_err, rot_err[0]


def predict_one_pose(p2d:np.ndarray, 
                     p3d:np.ndarray, 
                     intrinsic,
                     threshold=0.5)->dict:
    '''
    Input for poselib must follow format
    - p2d: list of np.array with shape (2, 1)
    - p3d: list of np.array with shape (3, 1)
    - camera as dictionary
    '''
    #uncertainty = 1/(1+100*np.abs(p3d[3, :]))
    #uncertainty = [True if tmpc >= threshold else False for tmpc in uncertainty]
    p2d = p2d + 0.5
    num_extracted_points = len(p2d)
    #p2d = p2d[uncertainty, :]
    start = time.time()
    p2d = [i for i in p2d] 
    #p3d = p3d[:, :3]
    p3d = p3d[:3, :]
    #p3d = p3d.T[uncertainty, :]
    p3d = p3d.T[:, :3]
    p3d = [i for i in p3d]
    pose, info = poselib.estimate_absolute_pose(
            p2d,
            p3d,
            intrinsic,
            {'max_reproj_error': 12.0},
            {}
            )
    end = time.time()
    est_time = end - start
    num_inliers = info['num_inliers']
    num_points = len(info['inliers'])
    return {
            "num_extracted_points": num_extracted_points,
            "pose": pose,
            "est_time": est_time,
            "num_inliers": num_inliers,
            "num_points": num_points
            }


class PL_SCR(pl.LightningModule):
    default_cfg = {

            }
    def __init__(self, 
                 cfg,
                 args):
        super().__init__()
        
        self.cfg    = {**self.default_cfg, **cfg}
        self.args   = args
        self.pretrained_ckpt = args.pretrained_ckpt
        self.init_lr    = self.args.init_lr
        self.batch_size = self.args.batch_size
        self.n_iter     = self.args.n_iter
        self.n_epoch    = self.args.n_epoch
        self.len_data   = self.args.len_data
        self.data_mode  = self.args.data_mode
        self.pose_mode  = self.args.pose_mode


        model = get_model(args.model, None)
        self.model = model(config ={})
        #self.loss = EuclideanLoss()
        loss_cls = get_loss(args.loss)
        self.loss = loss_cls()

        if self.pretrained_ckpt is not None:
            loguru_logger.info("Loading model and optimizer from checkpoint")
            if os.path.exists(args.pretrained_ckpt):
                checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state'])

        self.start_time = None
        self.best_loss  = np.inf
        self.increase_count = 0


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.args.init_lr, 
                                     eps=1e-8,
                                     betas=(0.9, 0.999))
        if self.pretrained_ckpt is not None:
            loguru_logger.info("Loading optimizer from checkpoint")
            if os.path.exists(self.pretrained_ckpt):
                checkpoint = torch.load(self.pretrained_ckpt, map_location='cpu')
                if 'optimizer_state' in checkpoint.keys():
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                else:
                    loguru_logger.info("Optimizer does not exist in state dict, \
                                       loading the default optimizer")
        return optimizer


    def training_step(self, batch, batch_idx):
        if self.data_mode == 'dense':
            coord_pred = self.model(batch['image'])
            coord_loss = self.loss(coord_pred, batch['coord'], batch['mask'])
            self.log("train_loss", coord_loss, 
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True)
            return coord_loss
        elif self.data_mode == 'sparse':
            output = self.model(batch)
            coord_loss = self.loss(output, batch)
            self.log("train_loss", coord_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True)
            return coord_loss
            


    def on_train_epoch_start(self):
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        lr = adjust_lr(optimizer, 
                       self.init_lr,
                       (self.current_epoch)*np.ceil(self.len_data/self.batch_size),
                       self.n_iter)
        if lr != current_lr:
            loguru_logger.info(
                    "Adjust learning rate from {} to {}".format(
                                                                current_lr, 
                                                                lr
                ))

    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        if self.current_epoch % int(np.floor(self.args.n_epoch / 5.)) == 0:
            loguru_logger.info("Saving model at epoch {}".format(self.current_epoch))
            save_state(self.args.save_path, 
                       self.current_epoch, 
                       self.model,
                       optimizer,
                       f'model_epoch-{self.current_epoch}.pkl')
        current_loss = self.trainer.callback_metrics.get('train_loss_epoch')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.increase_count = 0
            loguru_logger.info(f"New best loss: {self.best_loss:.4f}. Saving model...")
            save_state(self.args.save_path, 
                       self.current_epoch, 
                       self.model,
                       optimizer,
                       f'model_best_loss.pkl')
        else:
            self.increase_count += 1
            loguru_logger.info(f"Loss increased for {self.increase_count} consecutive epochs")

        #if self.increase_count >= 10:
        #    loguru_logger.info(f"Stopping training: Loss increased for 10 consecutive epochs")
        #    self.trainer.should_stop = True
            
    
    def on_test_epoch_start(self):
        # Get the intrinsic from data
        self.rot_err_list       = []
        self.transl_err_list    = []
        if self.data_mode == "dense":
            x = np.linspace(4, 640-4, 80) + 106 * (self.args.dataset == 'Cambridge')
            y = np.linspace(4, 480-4, 60)
            xx, yy = np.meshgrid(x, y)
            self.pcoord = np.concatenate((np.expand_dims(xx,axis=2), 
                    np.expand_dims(yy,axis=2)), axis=2)

            if self.args.dataset == '7Scenes':
                from datasets_pl.se7scene_pl import INTRINSIC_COLOR, CAMERA
                self.camera = CAMERA
            elif self.args.dataset == 'Cambridge':
                from datasets_pl.cambridge_pl import INTRINSIC_COLOR
            elif self.args.dataset == '12Scenes' or self.args.dataset == '12Scenes_depth':
                from datasets_pl.tw2scenes_pl import INTRINSIC_COLOR
            else:
                loguru_logger.info("Cannot retrieve intrinsic, check dataset name")
            self.solver = pnpransac(INTRINSIC_COLOR[0, 0],
                                    INTRINSIC_COLOR[1, 1],
                                    INTRINSIC_COLOR[0, 2],
                                    INTRINSIC_COLOR[1, 2])
            loguru_logger.info("Intialize pnp solver successfully")
            

    def test_step(self, batch, batch_idx):
        if self.data_mode == "dense":
            if self.args.dataset == 'Cambridge':
                batch['image'] = batch['image'][:,:,:,106:106+640]
            coord = self.model(batch['image'])
            coord = coord.cpu().data.numpy()[0, :, :, :]
            coord = np.transpose(coord, (1, 2, 0))

            coord   = np.ascontiguousarray(coord)
            pcoord  = np.ascontiguousarray(self.pcoord)
            p3d     = coord.reshape(-1, 3)
            p2d     = pcoord.reshape(-1, 2)
            pose_gt = batch['pose'].cpu().data.numpy()[0, :, :]
            #import pdb; pdb.set_trace()
            if self.pose_mode == "hscnet":
                rot, transl = self.solver.RANSAC_loop(p2d.astype(np.float64),
                                                      p3d.astype(np.float64), 
                                                    256)
                pose_est = np.eye(4)
                pose_est[0:3, 0:3] = cv2.Rodrigues(rot)[0].T
                pose_est[0:3, 3] = -np.dot(pose_est[0:3, 0:3], transl)

            elif self.pose_mode == "poselib":
                p3d = [i.reshape(3, 1) for i in p3d]
                p2d = [i.reshape(2, 1) for i in p2d]
                pose, info = poselib.estimate_absolute_pose(
                        p2d,
                        p3d,
                        self.camera,
                        {'max_reproj_error': 15.0},
                        {}
                        )
                r_pl = qvec2rotmat(pose.q)
                t_pl = pose.t
                pose_est = np.eye(4)
                pose_est[0:3, 0:3] = r_pl.T
                pose_est[0:3, 3]   = -np.dot(pose_est[0:3, 0:3], t_pl)

            transl_err, rot_err = get_pose_err(pose_gt, pose_est)
            self.rot_err_list.append(rot_err)
            self.transl_err_list.append(transl_err)
            
            loguru_logger.info(f'Pose error: {transl_err}m {rot_err}\u00b0')
        elif self.data_mode == "sparse":
            img_name    = batch['name'][0]
            intrinsic   = batch['sfm_intrinsic']
            pose_gt     = batch['sfm_pose'].squeeze().detach().cpu().numpy()
            # Minor mistake when saving data
            intrinsic['width'], intrinsic['height'] = intrinsic['height'], intrinsic['width']
            for k, v in intrinsic.items():
                if isinstance(v, list):
                    intrinsic[k] = v[0]
                elif k == 'params':
                    intrinsic[k] = v.squeeze().tolist()
                else:
                    intrinsic[k] = v.item()

            output      = self.model(batch)
            p2d         = output['keypoints'][0]
            p3d         = output['scenecoord']
            p2d         = p2d.squeeze().detach().cpu().numpy()
            p3d         = p3d.squeeze().detach().cpu().numpy()
            pose_output = predict_one_pose(p2d, 
                                           p3d,
                                           intrinsic)
            pose        = pose_output['pose']
            est_time    = pose_output['est_time']
            num_inliers = pose_output['num_inliers']
            num_points  = pose_output['num_points']
            num_extracted_points = pose_output['num_extracted_points']

            R           = qvec2rotmat(pose.q)
            t           = pose.t

            R_gt        = pose_gt[:3, :3]
            t_gt        = pose_gt[:3, 3]

            e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
            self.rot_err_list.append(e_R)
            self.transl_err_list.append(e_t)
            loguru_logger.info(f'Pose error: {e_t}m {e_R}\u00b0')

    def on_test_epoch_end(self):
        results = np.array([self.transl_err_list, self.rot_err_list]).T
        scene = self.args.scene 
        if '/' in scene:
            scene = scene.replace('/', '-')
        np.savetxt(os.path.join(self.args.save_path, 
                                'pose_err_{}_{}_{}.txt'.format(self.args.dataset,
                                                               scene,
                                                               self.args.model)), results)

        loguru_logger.info('Median pose error: {}m, {}\u00b0'.format(np.median(results[:, 0]),
                                                                     np.median(results[:, 1])))
        loguru_logger.info('Accuracy: {}%'.format(np.sum((results[:, 0] <= 0.05) 
                                                  *(results[:, 1]<=5)) * 1. / len(results)*100))



