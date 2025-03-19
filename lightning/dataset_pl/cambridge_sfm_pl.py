from pathlib import Path 
from tqdm import tqdm 
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .utils import load_img, load_pose_sfm, load_intrinsic_sfm

# With indoor6 dataset, we need to perform resize and keep the scale 
# to fix keypoint position

class CambridgeSfMDataset(Dataset):
    default_cfg = {
            "grayscale"     : True,
            "interpolation" : 'cv2_area',
            "resize_force"  : True,
            "resize_max"    : 1024
            }
    def __init__(self, cfg:dict={},
                 scene:str='GreatCourt',
                 root:str=None,
                 split:str='train')->None:
        self.cfg = {**self.default_cfg, **cfg}
        self.preprocess_cfg = {
                "grayscale"     : self.cfg["grayscale"],
                "interpolation" : self.cfg["interpolation"],
                "resize_force"  : self.cfg["resize_force"],
                "resize_max"    : self.cfg["resize_max"]
                }
        self.root               = Path(root)
        self.split              = split
        self.scene              = scene

        self.dataset_dir        = self.root / "cambridge" / self.scene / self.split
        self.img_dir            = self.dataset_dir / "images"
        self.sfm_intrinsics_dir = self.dataset_dir / "sfm_intrinsics"
        self.sfm_poses_dir      = self.dataset_dir / "sfm_poses"
        self.sfm_keypoints_dir  = self.dataset_dir / "sfm_keypoints"
        self.sfm_sc_dir         = self.dataset_dir / "sfm_scenecoord"
        self.sfm_sc_mask_dir    = self.dataset_dir / "sfm_scenecoord_mask"
        
        self.img_files = os.listdir(self.img_dir)
        self.img_files = [self.img_dir / f for f in self.img_files]


        self.img_transform = transforms.Compose([
            transforms.ToTensor()
            ])

    def __len__(self)->int:
        return len(self.img_files)


    def __getitem__(self, idx:int)->dict:
        #import pdb; pdb.set_trace()
        image_name = str(self.img_files[idx]).split('/')[-1]
        image_dict = load_img(self.img_files[idx], self.preprocess_cfg)
        self.preprocess_cfg['grayscale'] = True
        image_dict_gray = load_img(self.img_files[idx], 
                                    self.preprocess_cfg)

        self.preprocess_cfg['grayscale'] = False
        image = self.img_transform(image_dict['image'])
        image_gray = self.img_transform(image_dict_gray['image'])

        sfm_pose_f = self.sfm_poses_dir / image_name.replace('png', 
                                                            'pose.npy')
        sfm_intrinsic_f = self.sfm_intrinsics_dir / image_name.replace('png', 
                                                                       'intrinsic.pkl')
        sfm_keypoints_f = self.sfm_keypoints_dir / image_name.replace('png',
                                                                      'keypoints.npy')
        sfm_sc_f        = self.sfm_sc_dir / image_name.replace('png',
                                                               'scenecoord.npy')
        sfm_sc_mask_f   = self.sfm_sc_mask_dir / image_name.replace('png',
                                                                    'scenecoord_mask.npy')

        sfm_pose                = torch.from_numpy(load_pose_sfm(sfm_pose_f))

        if self.split == 'test':
            sfm_intrinsic           = load_intrinsic_sfm(sfm_intrinsic_f)
            return {
                    "name"          : image_name,
                    "image"         : image,
                    'image_gray'    : image_gray,
                    "sfm_pose"      : sfm_pose,
                    "sfm_intrinsic" : sfm_intrinsic,
                    "scale"         : image_dict['scale']
                    }
        else:
            sfm_intrinsic, radial   = load_intrinsic_sfm(sfm_intrinsic_f, type="mat")
            sfm_intrinsic           = torch.from_numpy(sfm_intrinsic)
            sfm_sc                  = torch.from_numpy(np.load(sfm_sc_f))
            sfm_keypoints           = torch.from_numpy(np.load(sfm_keypoints_f))
            sfm_sc_mask             = torch.from_numpy(np.load(sfm_sc_mask_f))
            return {"name"          : image_name,
                    "image"         : image, 
                    "image_gray"    : image_gray,
                    "sfm_pose"      : sfm_pose, 
                    "sfm_intrinsic" : sfm_intrinsic,
                    "sfm_keypoints" : sfm_keypoints,
                    "sfm_sc"        : sfm_sc,
                    "sfm_sc_mask"   : sfm_sc_mask,
                    "scale"         : image_dict['scale']
                    }

if __name__ == "__main__":
    cfg = {}
    root = Path("/home/thuan/Hoang_Workspace/datasets/sparse/")
    dataset = Indoor6SfMDataset(cfg=cfg,
                                  scene='GreatCourt',
                                  root=root)
    
    # Test if it load correctly and also the loading speed
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=0,
                            shuffle=True)

    for batch_idx, data in enumerate(tqdm(dataloader)):
        #print(batch_idx)
        continue

