# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import argparse

project_path = os.getcwd()
print(project_path)
if project_path not in sys.path:
  sys.path.append(project_path)

from basicsr.utils import scandir
#from basicsr.utils.create_lmdb import create_lmdb_for_gopro
from basicsr.utils.lmdb_util import make_lmdb_from_imgs


def prepare_keys(folder_path, suffix='png'):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix=suffix, recursive=False)))
    keys = [img_path.split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


def create_lmdb_for_gopro():
    '''
    megi docstringg
    '''
    folder_path = './datasets/GoPro/train/blur_crops'
    lmdb_path = './datasets/GoPro/train/blur_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/GoPro/train/sharp_crops'
    lmdb_path = './datasets/GoPro/train/sharp_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    #folder_path = './datasets/GoPro/test/target'
    #lmdb_path = './datasets/GoPro/test/target.lmdb'

    #img_path_list, keys = prepare_keys(folder_path, 'png')
    #make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    #folder_path = './datasets/GoPro/test/input'
    #lmdb_path = './datasets/GoPro/test/input.lmdb'

    #img_path_list, keys = prepare_keys(folder_path, 'png')
    #make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_fragments(dataset_name:str, 
                     create_for_train:bool=True,
                     create_for_test:bool=False):
    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3
    
    if create_for_train:
        opt['input_folder'] = f'./datasets/{dataset_name}/train/input'
        opt['save_folder'] = f'./datasets/{dataset_name}/train/blur_crops'
        opt['crop_size'] = 512
        opt['step'] = 256
        opt['thresh_size'] = 0
        extract_subimages(opt)

        opt['input_folder'] = f'./datasets/{dataset_name}/train/target'
        opt['save_folder'] = f'./datasets/{dataset_name}/train/sharp_crops'
        opt['crop_size'] = 512
        opt['step'] = 256
        opt['thresh_size'] = 0
        extract_subimages(opt)

    if create_for_test:
        opt['input_folder'] = f'./datasets/{dataset_name}/test/input'
        opt['save_folder'] = f'./datasets/{dataset_name}/test/blur_crops'
        opt['crop_size'] = 512
        opt['step'] = 256
        opt['thresh_size'] = 0
        extract_subimages(opt)

        opt['input_folder'] = f'./datasets/{dataset_name}/test/target'
        opt['save_folder'] = f'./datasets/{dataset_name}/test/sharp_crops'
        opt['crop_size'] = 512
        opt['step'] = 256
        opt['thresh_size'] = 0
        extract_subimages(opt)



def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(
            worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2',
                                '').replace('x3',
                                            '').replace('x4',
                                                        '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, c = img.shape
    else:
        raise ValueError(f'Image ndim should be 2 or 3, but got {img.ndim}')

    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'],
                         f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    def str2bool(v):
        return v.lower() in ('yes', 'true', 't', '1')
    
    parser.add_argument('--dataset_name', type=str, help='The name of the dataset for which to create crops', default='GoPro')
    parser.add_argument('--create_train', type=str2bool, help='True if question to create crops for training set, False otherwise', default=True)
    parser.add_argument('--create_test', type=str2bool, help='True if question to create crops for test set, False otherwise', default=False)

    args = parser.parse_args()

    create_fragments(dataset_name=args.dataset_name,
                     create_for_train=args.create_train,
                     create_for_test=args.create_test)
    
    #create_lmdb_for_gopro()
