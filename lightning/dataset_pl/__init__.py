from .se7scene_pl import Se7ScenesDataset
from .cambridge_pl import CambridgeDataset
from .se7scene_sfm_pl import Se7ScenesSfMDataset
from .tw2scenes_pl import TwelveScenesDataset
from .indoor6_sfm_pl import Indoor6SfMDataset
from .cambridge_sfm_pl import CambridgeSfMDataset
from .tw2scenes_depth_pl import TwelveScenesDepthDataset

def get_dataset(name):
    return {
            '7Scenes'       : Se7ScenesDataset,
            '7Scenes_SfM'   : Se7ScenesSfMDataset,
            'Cambridge'     : CambridgeDataset,
            'Cambridge_SfM' : CambridgeSfMDataset,
            '12Scenes'      : TwelveScenesDataset,
            '12Scenes_depth': TwelveScenesDepthDataset,
            'Indoor6_SfM'   : Indoor6SfMDataset,
            }[name]
