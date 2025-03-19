import torch 
import torch.nn as nn

import os 
import sys 
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from .sg_attn import MLP
# TODO: Add other features for comparison: SIFT


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    default_config = {
            "descriptor_dim": 256,
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": -1,
            "remove_borders": 4,
            }
    def __init__(self, config={}):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1) # useless.
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0) # useless.

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, self.config['descriptor_dim'], 
                                kernel_size=1, 
                                stride=1, padding=0)

        self.load_state_dict(torch.load('/media/slam/Data1/Hoang_workspace/erinyes_clone/erinyes/src/common/weights/superpoint_v1.pth'))
        print('loaded pre-trained SuperPoint.')

    def forward(self, data:dict):
        # encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        # Compute dense keypoint score
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])
        
        if 'sfm_keypoints' not in data.keys():
            # Extract keypoints
            keypoints   = [
                    torch.nonzero(s > self.config['keypoint_threshold'])
                    for s in scores
                    ]
            scores      = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]
            # Discard keypoints near image borders
            keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

            # Keep the k keypoints with highest score
            if self.config['max_keypoints'] >= 0:
                keypoints, scores = list(zip(*[
                    top_k_keypoints(k, s, self.config['max_keypoints'])
                    for k, s in zip(keypoints, scores)]))

            # Convert (h, w) to (x, y)
            keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        else:
            keypoints = data['sfm_keypoints']
            keypoints -= 0.5
            keypoints = [keypoints.squeeze().float()]


        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)

        descriptors = [sample_descriptors(k[None], d[None], 8)[0] 
                       for k, d in zip(keypoints, descriptors)]
        return {
                "keypoints"     : keypoints,
                "scores"        : scores,
                #"fmap"          : descriptors
                "descriptors"   : torch.stack(descriptors, dim=0)
                }


class SPSparseBaseline(nn.Module):
    default_conf = {
            "d_model": 256,
            "nhead": 4,
            "nlayer": 4,
            "layer_names": ['self']*4
            }
    def __init__(self, config):
        """
        Simplest SCR model, taken features and descriptors
        from the SP then pass over MLP to predict
        """
        super().__init__()
        self.conf = {**self.default_conf, **config}
        self.encoder = SuperPoint(config={
            'nms_radius'    : 3,
            'max_keypoints' : 2048
            })

        self.relu = nn.ReLU(inplace=True)
        # For sparse, we need to freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.mapping        = MLP([self.conf['d_model'], 512, 512, 1024, 1024, 512, 3])


    def forward(self, data:dict):
        x           = self.encoder(data)
        keypoints   = x['keypoints']
        scores      = x['scores']
        desc        = x['descriptors']
        
        # Stack list back to one tensor
        sparse_sc   = self.mapping(desc)
        return {
                'keypoints': keypoints,
                'scenecoord': sparse_sc
                }
        


if __name__ == "__main__":
    sys.path.append('..')
    from lightning.datasets_pl.se7scene_sfm_pl import Se7ScenesSfMDataset
    root = Path("/media/slam/Data1/Hoang_workspace/erinyes_clone/erinyes/dataset/7scenes/")
    model = SPSGSparseBaseline(config={})

    # Testing output 
    #input_ = torch.randn(1, 1, 480, 640)
    #keypoints = torch.randn(1, 2, 500)
    #data = {'image': input_,
    #        #'sfm_keypoints': keypoints
    #        }
    #out = model(data)

    dataset = Se7ScenesSfMDataset(cfg={},
                                  scene='chess',
                                  root=root)
    dataloader = DataLoader(dataset, 
                            batch_size=1,
                            num_workers=0,
                            shuffle=False)

    total_param = sum(p.numel() for p in model.parameters())
    print("Number of param", total_param)
    
    # Test with real dataset to check keypoints consistency
    for batch_idx, data in enumerate(tqdm(dataloader)):
        import pdb; pdb.set_trace()
        kp_bak = data.pop('sfm_keypoints')
        output = model(data)

        print(output['keypoints'])
        print(kp_bak)
        exit()







