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

INTRINSIC_COLOR = np.array([[572.0, 0.0, 320.0],
                            [0.0, 572.0, 240.0],
                            [0.0, 0.0, 1.0]])

INTRINSIC_COLOR_INV = np.linalg.inv(INTRINSIC_COLOR)

TRANSL = [[0,-20,0],[0,-20,0],[20,0,0],[20,0,0],[25,0,0],
        [20,0,0],[-20,0,0],[-25,5,0],[-20,0,0],[-20,-5,0],[0,20,0],
        [0,20,0]]

SUFFIXES_MAPPING = {
        'color': '.color.jpg',
        'pose' : '.pose.txt',
        'depth': '.depth.png',
        'label': '.label.png',
        'coord': '.coord.npy',
        'mask' : '.mask.npy'
        }

SCENES = [
        'apt1/kitchen',
        'apt1/living',
        'apt2/bed',
        'apt2/kitchen',
        'apt2/living',
        'apt2/luke',
        'office1/gates362',
        'office1/gates381',
        'office1/lounge',
        'office1/manolis',
        'office2/5a',
        'office2/5b'
        ]

class TwelveScenesDataset(data.Dataset):
    default_cfg = {
            "augmentation"  : False,
            "grayscale"     : True,
            }
    def __init__(self, cfg:dict={},
                 scene      :str='apt2/bed',
                 meta_root  :str="../",
                 img_root   :str='../',
                 split      :str='train'):
        """

        """

        self.cfg        = {**self.default_cfg, **cfg}
        self.meta_root  = meta_root
        self.scene      = scene 
        self.img_root   = img_root 
        self.split      = split

        assert self.scene in SCENES

        self.meta_root  = os.path.join(self.meta_root, '12Scenes')
        self.img_root   = os.path.join(self.img_root, '12Scenes')

        with open(os.path.join(self.meta_root, self.scene, '{}{}'.format(self.split,
                                                             '.txt')), 'r') as f:
            self.frames = f.readlines()

        self.centers = np.load(os.path.join(self.meta_root,
                                            scene,
                                            'centers.npy'))


    def __len__(self):
        return len(self.frames)


    def __getitem__(self, index):
        frame = self.frames[index].rstrip('\n')

        img_path    = os.path.join(self.img_root,
                                self.scene,
                                'data',
                                '{}{}'.format(frame, SUFFIXES_MAPPING['color']))

        label_path  = os.path.join(self.meta_root,
                                   self.scene,
                                   'data',
                                   '{}{}'.format(frame, SUFFIXES_MAPPING['label']))

        pose_path   = img_path.replace(SUFFIXES_MAPPING['color'],
                                       SUFFIXES_MAPPING['pose'])
        coord_path  = label_path.replace(SUFFIXES_MAPPING['label'],
                                         SUFFIXES_MAPPING['coord'])
        mask_path   = label_path.replace(SUFFIXES_MAPPING['label'],
                                         SUFFIXES_MAPPING['mask'])

        if self.cfg['grayscale']:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            image       = cv2.imread(img_path)
            image       = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image       = cv2.resize(image, (640, 480))

        pose        = np.loadtxt(pose_path)

        if self.split == 'test':
            if self.cfg['grayscale']:
                image = np.expand_dims(image, axis=0)
            else:
                image = image.transpose(2, 0, 1)
            image = image / 255.
            image = image * 2. - 1
            return {
                    'image' : torch.from_numpy(image).float(),
                    'pose'  : torch.from_numpy(pose).float()
                    }

        coord       = np.load(coord_path)
        mask        = np.load(mask_path)

        if self.cfg['augmentation']:
            centers = self.centers
            lbl     = cv2.imread(label_path, -1)
            ctr_coord = centers[np.reshape(lbl,(-1))-1,:]
            ctr_coord = np.reshape(ctr_coord,(480,640,3)) * 1000
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
        coord = coord[4::8,4::8,:]
        mask = mask[4::8,4::8].astype(np.float16)
        coord   = coord.transpose(2, 0, 1)
        coord   = coord / 1000.


        image = torch.from_numpy(image).float()
        coord = torch.from_numpy(coord).float()
        mask  = torch.from_numpy(mask).float()
        pose  = torch.from_numpy(pose).float()

        return {
                'image' : image,
                'pose'  : pose,
                'coord' : coord,
                'mask'  : mask
                }

if __name__ == "__main__":
    dataset = TwelveScenesDataset(cfg={'augmentation':False},
                           scene='apt1/kitchen',
                           meta_root="/home/thuan/Hoang_Workspace/datasets/hscnet/",
                           img_root="/home/thuan/Hoang_Workspace/datasets/",
                           split='train')

    dataloader = data.DataLoader(dataset,
                                 batch_size=1,
                                 num_workers=0,
                                 shuffle=False)

    for batch_idx, (data) in enumerate(tqdm(dataloader)):
        continue



        

