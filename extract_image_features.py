# From Perez et al, AAAI 2018
# https://github.com/ethanjperez/film/blob/master/scripts/extract_features.py

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, os, json
import h5py as h5
import numpy as np
from scipy.misc import imread, imresize
from tqdm import tqdm
import pudb
import torch
import torchvision
import cv2
import torch.nn as nn
import pickle
from pathlib import Path
from PIL import Image
from torchvision import transforms
from added_func import *

parser = argparse.ArgumentParser()
parser.add_argument('--max_images', default=None, type=int)
# parser.add_argument('--input_h5_file', required=True)
parser.add_argument('--data_dir', default='../dataset')
parser.add_argument('--output_h5_file', default='image_features.h5')

parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

parser.add_argument('--model', default='resnet101')
parser.add_argument('--model_stage', default=2, type=int)
# model_stage_1 --> 1200, 256, 56, 56
# model_stage 2 --> 1200, 512, 28, 28 # lets go with this
# model_stage 3 --> 1200, 1024, 14, 14
# model_stage 4 --> 1200, 2048, 7, 7
parser.add_argument('--batch_size', default=128, type=int)
# parser.add_argument('--data_size', default=1024, type=int)
parser.add_argument('--bw', default='No')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_model(args):
    if args.model.lower() == 'none':
        return None
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if not 'resnet' in args.model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    layers = [
        cnn.conv1,
        cnn.bn1,
        cnn.relu,
        cnn.maxpool,
    ]
    for i in range(args.model_stage):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(cnn, name))
    model = torch.nn.Sequential(*layers)

    

    model = nn.DataParallel(model)
    model.to(device)   
    model.eval()
    return model


def run_batch(cur_batch, model):
    if model is None:
        image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
        return image_batch / 255.  # Scale pixel values to [0, 1]

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).to(device)
    with torch.no_grad():
        image_batch = torch.autograd.Variable(image_batch)

    feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()

    return feats


def main(args):

    # input_h5_file = h5.File(args.input_h5_file, 'r')
    model = build_model(args)
    data_dir = args.data_dir
    # data_size = args.data_size

    image_data_h5 = h5.File('image_data.h5', 'w')

    with h5.File(args.output_h5_file, 'w') as f:
        for split in ['train', 'val']:
            feat_dset = None
            i0 = 0
            cur_batch = []
            # image_files = input_h5_file[split+'_ims']
            file_id_list = get_file_id_list(split)

            if split == 'train':
                rgb_dir = 'rgb_train'
            else:
                rgb_dir = 'rgb_test'

            extracted_file_id_list = []

            for file_id in file_id_list:
                color_path = Path(f"{data_dir}/{rgb_dir}/{file_id}.png")
                # print(color_path)
                if not (color_path).exists():
                    continue
                else:
                    extracted_file_id_list.append(file_id)

            extracted_file_id_list = extracted_file_id_list
            print(len(extracted_file_id_list))
            image_files, loaded_file_ids = load_images(split, extracted_file_id_list)

            image_data_h5.create_dataset(f'image_{split}', data=image_files)
            image_data_h5.create_dataset(f'file_ids_{split}', data=loaded_file_ids)

            for i, img in tqdm(enumerate(image_files)):
                # converting the image to gray
                # print(i)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.stack((img,)*3, -1)

                img = img.transpose(2, 0, 1)[None]
                cur_batch.append(img)
                if len(cur_batch) == args.batch_size:
                    feats = run_batch(cur_batch, model)
                    if feat_dset is None:
                        N = len(image_files)
                        _, C, H, W = feats.shape
                        feat_dset = f.create_dataset(split+'_features', (N, C, H, W),
                                           dtype=np.float32)
                    i1 = i0 + len(cur_batch)
                    feat_dset[i0:i1] = feats
                    i0 = i1
                    cur_batch = []
                    # print(len(cur_batch))
            if len(cur_batch) > 0:
                feats = run_batch(cur_batch, model)
                if feat_dset is None:
                    N = len(image_files)
                    _, C, H, W = feats.shape
                    feat_dset = f.create_dataset(split+'_features', (N, C, H, W),
                                       dtype=np.float32)
                i1 = i0 + len(cur_batch)

                feat_dset[i0:i1] = feats

    image_data_h5.close()
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



