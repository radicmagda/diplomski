import importlib
import os
import os.path as osp
import torch
from tqdm import tqdm
from copy import deepcopy
from basicsr.utils import get_root_logger, tensor2img, imwrite
from basicsr.models.base_model import BaseModel
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class DummyModel(BaseModel):
    def __init__(self, opt):
        super(DummyModel, self).__init__(opt)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)

    def test(self):
        # Dummy "prediction": just use the input as output
        self.output = self.lq

    def get_current_visuals(self):
        return {'lq': self.lq.detach(), 'result': self.output.detach(), 'gt': self.gt.detach()}

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        # Progress bar for single-process execution
        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # Tentative for out-of-GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], img_name,
                            f'{img_name}_{current_iter}.png'
                        )
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], img_name,
                            f'{img_name}_{current_iter}_gt.png'
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png'
                        )
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_gt.png'
                        )

                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # Calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        # Collect and log metrics
        if with_metrics:
            metrics_dict = {}
            for key, value in self.metric_results.items():
                metrics_dict[key] = value / cnt

            self._log_validation_metric_values(
                current_iter, dataloader.dataset.opt['name'], tb_logger, metrics_dict
            )

        return 0.

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict
