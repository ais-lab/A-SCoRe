import os
import sys
import random 
import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils import data
from .utils import (read_image, 
                   resize_image,
                   get_depth,
                   get_coord,
                   data_aug)
#sys.path.append('../..')
#from datasets.se7scene import Se7Scenes

# Consider where data come from 
# 1. Image: One source only 
# 2. Pose: 
#   2.1 GT Pose
#   2.2 SfM poses
# 3. Depth:
#   3.1 GT Depth 
#   3.2 Rendered depth
# 4. Scene coord
#   4.1 Calculate from depth at run run time 
#   4.2 Pre-computed from depth
# 5. Augmentation
#   5.1 DSAC Augmentation
#   5.2 Self Augmentation

INTRINSIC_COLOR = np.array([[525.0, 0.0, 320.0],
                            [0.0, 525.0, 240.0],
                            [0.0, 0.0, 1.0]])

INTRINSIC_DEPTH = np.array([[585.0, 0.0, 320.0],
                            [0.0, 585.0, 240.0],
                            [0.0, 0.0, 1.0]])

INTRINSIC_COLOR_INV = np.linalg.inv(INTRINSIC_COLOR)

INTRINSIC_DEPTH_INV = np.linalg.inv(INTRINSIC_DEPTH)

SCENES = ['chess', 
          'fire', 
          'heads', 
          'office', 
          'pumpkin',
          'redkitchen',
          'stairs']

SUFFIXES_MAPPING = {
        'color': '.color.png',
        'pose' : '.pose.txt',
        'depth': '.depth.png',
        'label': '.label.png',    # This is label for classification
        'coord': '.coord.npy',
        'mask' : '.mask.npy'
        }

CAMERA = {
        'model': 'SIMPLE_PINHOLE',
        'width': 640,
        'height': 480,
        'params': [525.0, 320.0, 240.0]
        }


class Se7ScenesDataset(data.Dataset):
    default_cfg = {
            "pose"          :"gt",   # gt/sfm
            "depth"         :None,  # gt/rendered/None
            "scenecoord"    :"pre-computed",    # pre-computed/runtime
            "grayscale"     :False,
            "resize"        :(),
            "augmentation"  :True,
            }
    def __init__(self, cfg:dict={}, 
                 scene      :str='chess',
                 meta_root  :str='../',
                 img_root   :str='../',
                 split      :str='train'):
        """
        scene       : Name of the scene, must be in SCENES
        meta_root   : Root directory of metadata, extracted from hscnet data
        img_root    : Root directory of images, directly extracted from 7Scenes 
        split       : train/test split
        """
        self.cfg        = {**self.default_cfg, **cfg}
        self.meta_root  = meta_root
        self.img_root   = img_root
        self.scene      = scene
        self.split      = split

        assert self.scene in SCENES

        self.meta_root  = os.path.join(self.meta_root, '7Scenes')
        self.img_root   = os.path.join(self.img_root, '7scenes')

        # Load the extrinsic calib
        self.calibration_extrinsics = np.loadtxt(os.path.join(self.meta_root,
                                                              'sensorTrans.txt'))

        # Center coordinate of the scene calculated beforehand
        self.scene_ctr  = np.loadtxt(os.path.join(self.meta_root,
                                                  self.scene,
                                                  'translation.txt'))
        self.centers    = np.load(os.path.join(self.meta_root,
                                               self.scene,
                                               'centers.npy'))

        # Load the list of training files
        with open(os.path.join(self.meta_root, '{}{}'.format(self.split,
                                                        '.txt')), 'r') as f:
            self.frames = f.readlines()
            self.frames = [frame for frame in self.frames \
                    if self.scene in frame]


    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index].rstrip('\n')
        _, seq_id, frame_id = frame.split(' ')

        img_path = os.path.join(self.img_root,
                                self.scene,
                                seq_id,
                                '{}{}'.format(frame_id, SUFFIXES_MAPPING['color']))

        label_path = os.path.join(self.meta_root,
                                  self.scene,
                                  seq_id,
                                  '{}{}'.format(frame_id, SUFFIXES_MAPPING['label']))

        image   = read_image(img_path, self.cfg['grayscale'])

        # Check to whether load pose sfm or gt
        if self.cfg['pose'] == 'gt':
            pose_path = img_path.replace(SUFFIXES_MAPPING['color'],
                                         SUFFIXES_MAPPING['pose'])
        else:
            pass

        pose = np.loadtxt(pose_path)
        pose[0:3, 3] = pose[0:3, 3] - self.scene_ctr

        if self.split == 'test':
            if self.cfg['grayscale']:
                image = np.expand_dims(image, axis=0)
            else:
                image = image.transpose(2, 0, 1)
            image = image / 255.
            image = image * 2. - 1.
            return {
                    'frame_id': frame_id,
                    'seq_id': seq_id,
                    'image' : torch.from_numpy(image).float(),
                    'pose'  : torch.from_numpy(pose).float(),
                    }

        lbl     = cv2.imread(label_path, -1)

        # Check to whether load gt depth or rendered depth
        if self.cfg['depth'] is None:
            pass
        elif self.cfg['depth'] == 'gt':
            depth_path = img_path.replace(SUFFIXES_MAPPING['color'],
                                          SUFFIXES_MAPPING['depth'])
            depth = cv2.imread(depth_path, -1)
            depth[depth==65535] = 0
            depth = depth * 1.0
            depth = get_depth(depth, self.calibration_extrinsics,
                              INTRINSIC_COLOR,
                              INTRINSIC_DEPTH_INV)

        # Check to load pre-computed coord or calculate at runtime
        if self.cfg['scenecoord'] == 'pre-computed':
            coord_path = label_path.replace(SUFFIXES_MAPPING['label'],
                                            SUFFIXES_MAPPING['coord'])

            mask_path = label_path.replace(SUFFIXES_MAPPING['label'],
                                           SUFFIXES_MAPPING['mask'])
            coord   = np.load(coord_path)
            mask    = np.load(mask_path)

        elif self.cfg['scenecoord'] == 'runtime':
            coord, mask = get_coord(depth, pose, 
                                    INTRINSIC_COLOR_INV)

        
        # Check to perform augmentation
        if self.cfg['augmentation']:
            centers = self.centers
            ctr_coord = centers[np.reshape(lbl, (-1))-1, :]
            ctr_coord = np.reshape(ctr_coord, (480, 640, 3)) * 1000
            #import pdb; pdb.set_trace()
            image, coord, ctr_coord, mask, lbl = data_aug(image,
                                                          coord,
                                                          ctr_coord,
                                                          mask,
                                                          lbl, 
                                                          self.cfg['augmentation'],
                                                          self.cfg['grayscale'])

        if self.cfg['grayscale']:
            image = np.expand_dims(image, axis=0)
        else:
            image = image.transpose(2, 0, 1)

        image = image / 255.
        image = image * 2. - 1.
        coord   = coord[4::8, 4::8,:]
        mask    = mask[4::8, 4::8].astype(np.float16)
        coord   = coord.transpose(2, 0, 1)
        coord   = coord / 1000.

        # Convert all to tensor type
        image = torch.from_numpy(image).float()
        coord = torch.from_numpy(coord).float()
        mask  = torch.from_numpy(mask).float()
        pose  = torch.from_numpy(pose).float()

        return {
                'frame_id': frame_id,
                'seq_id': seq_id,
                'image' : image,
                'pose'  : pose,
                'coord' : coord,
                'mask'  : mask
                }

if __name__ == "__main__":
    dataset = Se7ScenesDataset(cfg={'augmentation': False,
                                    'grayscale': False},
                               scene='chess',
                               meta_root="/media/slam/Storage500GB/Data/hscnet/data/",
                               img_root="/media/slam/Storage500GB/Data",
                               split="train")
    
    # Test for running speed
    dataloader = data.DataLoader(dataset, 
                                 batch_size=1,
                                 num_workers=0,
                                 shuffle=True)

    for batch_idx, (data) in enumerate(tqdm(dataloader)):
        #import pdb; pdb.set_trace()
        print(batch_idx)


    # Test for correctness 

    #root = "/media/slam/Storage500GB/Data/hscnet/data/"
    #img_folder = "/media/slam/Storage500GB/Data"
    #dataset_gt = Se7Scenes(root,
    #                       img_folder,
    #                       '7S',
    #                       'chess',
    #                       'train',
    #                       'scrnet',
    #                       False)
    #datum = dataset.__getitem__(0)
    #datum_gt = dataset_gt.__getitem__(0)
    #
    #import pdb; pdb.set_trace()
    #assert torch.equal(datum['coord'], datum_gt[1])
    #assert torch.equal(datum['mask'], datum_gt[2])
    #assert torch.equal(datum['image'], datum_gt[0])
    




