import os 
import sys 
from yacs.config import CfgNode as CN
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import pickle
from pathlib import Path
from loguru import logger as loguru_logger

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.read_write_model import read_model
from helper.utils import update_config_with_args, format_output_dir
from config.preprocess.se7scenes import get_cfg_defaults


class Se7scenesProcessing():
    default_conf = {}
    def __init__(self, cfg) -> None:
        self.dataset_dir = cfg.DATASET_DIR
        self.sfm_dir = cfg.SFM_DIR
        self.rendered_depth_dir = cfg.RENDERED_DEPTH_DIR
        self.scenes = cfg.SCENES
        self.use_depth = cfg.USE_DEPTH
        self.output = cfg.OUTPUT

    def read_scene_sfm(self, scene:str)->dict:
        assert self.dataset_dir is not None 
        assert self.sfm_dir is not None
        sfm_abs_path = os.path.join(self.sfm_dir, 
                                    scene)
        sfm_cams, sfm_images, sfm_scenecoord = read_model(sfm_abs_path)
        return {'cams': sfm_cams, 
                'images': sfm_images, 
                'scenecoord': sfm_scenecoord}

    def read_testlist(self, scene:str)->list:
        assert self.sfm_dir is not None
        testlist_abs_path = os.path.join(self.sfm_dir, 
                                         scene,
                                         "sfm_sift_full",
                                         "list_test.txt")
        if os.path.exists(testlist_abs_path):
            with open(testlist_abs_path, 'r') as f:
                testlist = f.read().rstrip().split('\n')
        else:
            loguru_logger.debug("Error testlist file not found !!!")
        return testlist


    def run(self):
        loguru_logger.info("Preprocessing 7scenes dataset")
        assert self.output is not None
        for scene in self.scenes:
            sfm_scenecoord_mean = np.zeros((3))
            count = 0
            loguru_logger.info(f"Working on scene {scene}")
            output_dir = format_output_dir(os.path.join(self.output, scene))
            loguru_logger.info("Reading SfM model")
            ref_sfm_sift = os.path.join(self.sfm_dir,
                                        scene,
                                        "sfm_sift_full")
            if self.use_depth:
                sfm_sp_sg = os.path.join(self.sfm_dir,
                                         scene,
                                         "sfm_superpoint+superglue+depth")
            else:
                sfm_sp_sg = os.path.join(self.sfm_dir,
                                         scene,
                                         "sfm_superpoint+superglue")
            testlist = self.read_testlist(scene)
            ref_sfm_sift = self.read_scene_sfm(ref_sfm_sift)
            sfm_sp_sg = self.read_scene_sfm(sfm_sp_sg)
            name2id_train = {image.name: i for i, image in sfm_sp_sg['images'].items()}
            for _, image in tqdm(ref_sfm_sift['images'].items()):
                #sfm_pose = f"{image.qvec} {image.tvec}"
                saved_name = image.name.replace('/', '-')
                sfm_pose = np.concatenate((image.qvec, image.tvec))
                sfm_camera = ref_sfm_sift['cams'][image.camera_id]
                sfm_camera = {
                        "model": sfm_camera.model,
                        "width": sfm_camera.width,
                        "height": sfm_camera.height,
                        "params": sfm_camera.params
                        }
                if image.name in testlist:
                    mode = "test"
                    sfm_keypoints = image.xys
                    sfm_scenecoord = np.stack([ref_sfm_sift['scenecoord'][ii].xyz if ii !=-1 else 
                                               np.array([0, 0, 0]) for ii in image.point3D_ids], 0)
                    sfm_scenecoord_mask = np.stack([1 if ii !=-1 else 
                                                    0 for ii in image.point3D_ids], 0)
                else:
                    mode = "train"
                    train_img = sfm_sp_sg['images'][name2id_train[image.name]]
                    sfm_keypoints = train_img.xys
                    sfm_scenecoord = np.stack([sfm_sp_sg['scenecoord'][ii].xyz if ii != -1 else 
                                               np.array([0, 0, 0]) for ii in train_img.point3D_ids], 0)
                    sfm_scenecoord_mask = np.stack([1 if ii != -1 else 
                                                    0 for ii in train_img.point3D_ids], 0)
                    sfm_scenecoord_mean += sfm_scenecoord.sum(0)
                    count += sfm_scenecoord_mask.sum()

                # Create symbolic link from original image 
                os.system(
                        'ln -s ' + 
                        self.dataset_dir + '/' + 
                        scene + '/' + 
                        image.name + ' ' + 
                        str(output_dir[mode]['images']) + '/' + 
                        saved_name)

                # Create symbolic link from the rendered depth from DSAC 
                os.system( 'ln -s ' + 
                        self.rendered_depth_dir + '/' + 
                        f'7scenes_{scene}' + '/' +
                        f'{mode}' + '/' +
                        "depth" + '/' +
                        image.name.replace('/', '_').replace('color.png', 'pose.depth.tiff').replace('-', "", 1) + ' ' + 
                        str(output_dir[mode]['rendered_depth']) + '/' + 
                        image.name.replace('/', '-').replace('color.png', 'pose.depth.tiff')
                        )

                # Save pose as numpy array 
                np.save(output_dir[mode]['sfm_poses'] / saved_name.replace('color.png', 'pose.npy'), 
                        sfm_pose)
                # Save camera using pickle
                with open(output_dir[mode]['sfm_intrinsics'] / saved_name.replace('color.png', 'intrinsic.pkl'), 'wb') as file: pickle.dump(sfm_camera, file)

                # Save sfm keypoints
                if sfm_keypoints is not None:
                    np.save(output_dir[mode]['sfm_keypoints'] / saved_name.replace('color.png', 'keypoints.npy'), 
                            sfm_keypoints)
                # Save sfm scene coord and mask
                if sfm_scenecoord is not None:
                    np.save(output_dir[mode]['sfm_scenecoord'] / saved_name.replace('color.png', 'scenecoord.npy'), 
                            sfm_scenecoord)
                if sfm_scenecoord_mask is not None:
                    np.save(output_dir[mode]['sfm_scenecoord_mask'] / saved_name.replace('color.png', 'scenecoord_mask.npy'), 
                            sfm_scenecoord_mask)

            sfm_scenecoord_mean /= count
            np.save(output_dir[mode]["root"] / f'{scene}_sfm_sc_mean.npy', sfm_scenecoord_mean)
                

def main(args):
    processor = Se7scenesProcessing(args)
    processor.run()

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    parser = ArgumentParser(description="Preprocess 7scenes dataset")
    parser.add_argument('--dataset_dir', type=str, 
                        default=None,
                        help="")
    parser.add_argument('--sfm_dir', type=str,
                        default=None,
                        help="")
    parser.add_argument('--use_depth', 
                        action='store_true')
    parser.add_argument('--rendered_depth_dir', type=str,
                        default=None)
    parser.add_argument('--output', type=str,
                        default=None)
    parser.add_argument('--scenes', type=str, 
                        nargs='*', 
                        default=None, 
                        help="List of  the scene want to preprocess")
    args = parser.parse_args()

    loguru_logger.info(f"Default config: \n{cfg}")
    update_config_with_args(cfg, args)
    loguru_logger.info(f"Updated config: \n{cfg}")

    main(cfg)

