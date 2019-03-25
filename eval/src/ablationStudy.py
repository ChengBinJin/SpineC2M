# ---------------------------------------------------------
# Tensorflow SpineC2M Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import cv2
import numpy as np
from utils import all_files_under, structural_similarity_index

class SliceExample:
    def __init__(self, name, measurement):
        self.name = name
        self.measurement = measurement

    def __repr__(self):
        return repr((self.name, self.measurement))


def read_imgs_and_measurement(target=None):
    slice_list = []

    # read gt image addresses
    gt_names = all_files_under('../gt', extension='.jpg')

    for idx in range(len(gt_names)):
        if np.mod(idx, 300) == 0:
            print('idx: {}'.format(idx))

        img_name = os.path.basename(gt_names[idx])

        # read gt and prediction image
        gt_img = cv2.imread(gt_names[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        pred_img = cv2.imread(os.path.join('../{}'.format(target), img_name),
                              cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # calculate measurement
        measure_value = structural_similarity_index(pred_img, gt_img)
        slice_list.append(SliceExample(img_name, measure_value))

    return slice_list


def main(folders, size, margin=10):
    save_folder = 'ablation_study'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    slice_list = read_imgs_and_measurement(target=folders[-2])
    new_slice_list = sorted(slice_list, key=lambda example: example.measurement, reverse=True)

    # save imgs
    for idx,  example in enumerate(new_slice_list):
        if idx % 300 == 0:
            print('idx: {}, img name: {}, SSIM:{}'.format(idx, example.name, example.measurement))

        # Initialize white canvas
        canvas = 255 * np.ones((size[0], len(folders) * size[1] + (len(folders) - 1) * margin), dtype=np.uint8)

        for idx_term, data_path in enumerate(folders):
            img = cv2.imread(os.path.join('../{}'.format(data_path), example.name), cv2.IMREAD_GRAYSCALE)
            canvas[:, idx_term * margin + idx_term * size[1]: idx_term * margin + (idx_term + 1) * size[1]] = img

        cv2.imwrite(os.path.join(save_folder, str(idx).zfill(4) + '_' + example.name), canvas)


if __name__ == '__main__':
    target_folders = ['ct', 'DC2Anet_20190301-1512', 'DC2Anet_20190228-1700', 'DC2Anet_20190222-0737',
                      'DC2Anet_20190222-1039', 'DC2Anet_20190225-1341', 'DC2Anet_20190223-1334', 'gt']
    img_size = (300, 200, 1)
    margin = 5

    main(folders=target_folders, size=img_size, margin=margin)
