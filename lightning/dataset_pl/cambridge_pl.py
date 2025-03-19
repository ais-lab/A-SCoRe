import os 
import random 
import numpy as np
import cv2 
import torch
from torch.utils import data 
from .utils import (
        read_image,
        data_aug
        )


INTRINSIC_COLOR = np.array([[744.375, 0.0, 426.0],
                            [0.0, 744.375, 240.0],
                            [0.0, 0.0, 1.0]])

SCENES = [
        'GreatCourt',
        'KingsCollege',
        'OldHospital',
        'ShopFacade',
        'StMarysChurch'
        ]

INTRINSIC_COLOR_INV = np.linalg.inv(INTRINSIC_COLOR)

SUFFIXES_MAPPING = {
        "color": ".png",
        "pose": ".pose.txt",
        "depth": ".depth.png",
        "label": ".label.png",
        "coord": ".coord.npy",
        "mask": ".mask.npy"
        }


class CambridgeDataset(data.Dataset):
    default_cfg = {

            }
    def __init__(self, cfg:dict={},
                 scene      :str='GreatCourt',
                 meta_root  :str='../',
                 img_root   :str="../",
                 split      :str="train"):
        self.cfg        = {**self.default_cfg, **cfg}
        self.aug        = self.cfg['augmentation']
        self.img_root   = os.path.join(img_root, "cambridge")
        self.scene      = scene 
        self.split      = split

        self.meta_root  = os.path.join(meta_root, 
                                       'Cambridge', 
                                       self.scene)

        assert self.scene in SCENES

        self.centers = np.load(os.path.join(self.meta_root, 'centers.npy'))

        with open(os.path.join(self.meta_root, '{}{}'.format(self.split, '.txt')), 'r') as f:
            self.frames = f.readlines()

    def __len__(self):
        return len(self.frames)


    def __getitem__(self, index):
        frame = self.frames[index].rstrip('\n')
        seq_id = frame.split('_')[0]


        img_path = os.path.join(self.img_root,
                                self.scene,
                                seq_id,
                                '{}{}'.format(frame.split('_')[1], '.png'))
        label_path = os.path.join(self.meta_root,
                                  self.split,
                                  '{}{}'.format(frame, '.label.png'))

        pose_path  = label_path.replace(SUFFIXES_MAPPING['label'], 
                                        SUFFIXES_MAPPING['pose'])
        coord_path = label_path.replace(SUFFIXES_MAPPING['label'], 
                                        SUFFIXES_MAPPING['coord'])
        mask_path = label_path.replace(SUFFIXES_MAPPING['label'], 
                                       SUFFIXES_MAPPING['mask'])

        if self.cfg['grayscale']:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (852, 480))
        pose = np.loadtxt(pose_path)

        if self.split == 'test':
            if self.cfg['grayscale']:
                img = np.expand_dims(img, axis=0)
            else:
                img = img.transpose(2, 0, 1)

            img = img / 255.
            img = img * 2. - 1
            return {
                    'image' : torch.from_numpy(img).float(),
                    'pose'  : torch.from_numpy(pose).float()
                    }
        coord = np.load(coord_path)
        mask  = np.load(mask_path)

        if self.cfg['augmentation']:
            lbl = cv2.imread(label_path,-1) 
            ctr_coord = self.centers[np.reshape(lbl,(-1))-1,:]
            ctr_coord = np.reshape(ctr_coord,(480,852,3)) * 1000

            img, coord, ctr_coord, mask, lbl = data_aug(img, coord, ctr_coord,
                    mask, lbl, self.aug, self.cfg['grayscale'])

        
        img_h, img_w = img.shape[0:2]
        th, tw = 480, 640
        x1 = random.randint(0, img_w - tw)
        y1 = random.randint(0, img_h - th)
        
        img = np.expand_dims(img, axis=-1) 
        img = img[y1:y1+th,x1:x1+tw,:]
        coord = coord[y1:y1+th,x1:x1+tw,:]
        mask = mask[y1:y1+th,x1:x1+tw]
        
        coord = coord[4::8,4::8,:]
        mask = mask[4::8,4::8].astype(np.float16)
        img = img / 255.
        img = img * 2. - 1.
        img = img.transpose(2, 0, 1)
        coord = coord.transpose(2, 0, 1)
        coord = coord / 1000.

        image = torch.from_numpy(img).float()
        coord = torch.from_numpy(coord).float()
        mask  = torch.from_numpy(mask).float()
        pose  = torch.from_numpy(pose).float()


        return {
                'name'  : frame,
                'image' : image,
                'pose'  : pose,
                'coord' : coord,
                'mask'  : mask
                }

if __name__ == "__main__":
    dataset = CambridgeDataset(cfg={'grayscale': True,
                                    'augmentation': False},
                               scene='GreatCourt',
                               meta_root='/mnt/Storage4TB/dataset/hscnet/data/',
                               img_root='/mnt/Storage4TB/dataset/',
                               split='train')
    dataloader = data.DataLoader(dataset,
                                 batch_size=1,
                                 num_workers=0,
                                 shuffle=False)

    from tqdm import tqdm
    for batch_idx, (data) in enumerate(tqdm(dataloader)):
        #import pdb; pdb.set_trace()
        continue



