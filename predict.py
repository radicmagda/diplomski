import torch
import numpy as np
import cv2
import tempfile
import matplotlib.pyplot as plt
from cog import BasePredictor, Path, Input, BaseModel

from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse


class Predictor(BasePredictor):
    def setup(self):

        opt_path_deblur = "options/test/GoPro/NAFNet-width64.yml"
        opt_deblur = parse(opt_path_deblur, is_train=False)
        opt_deblur["dist"] = False

        self.models = {
            "Image Debluring": create_model(opt_deblur),
        }

    def predict(
        self,
        task_type: str = Input(
            choices=[
                "Image Debluring",
            ],
            default="Image Debluring",
            description="Choose task type.",
        ),
        image: Path = Input(
            description="Input image. Stereo Image Super-Resolution, upload the left image here.",
        ),
    ) -> Path:

        out_path = Path(tempfile.mkdtemp()) / "output.png"

        model = self.models[task_type]
        img_input = imread(str(image))
        inp = img2tensor(img_input)
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        single_image_inference(model, inp, str(out_path))

        return out_path


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.0
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)


def single_image_inference(model, img, save_path):
    model.feed_data(data={"lq": img.unsqueeze(dim=0)})

    if model.opt["val"].get("grids", False):
        model.grids()

    model.test()

    if model.opt["val"].get("grids", False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals["result"]])
    imwrite(sr_img, save_path)