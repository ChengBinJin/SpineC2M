# ---------------------------------------------------------
# Tensorflow SpineC2M Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import cv2
import argparse
import numpy as np

import utils as utils

parser = argparse.ArgumentParser(description='')
parser.add_argument('--method', dest='method', default='mrgan', type=str,
                    help='select from [pix2pix|cyclegan|discogan|mrgan]')
parser.add_argument('--measure', dest='measure', default='ssim', type=str, help='select from [mae|rmse|psnr|ssim]')
parser.add_argument('--number', dest='number', default=20, type=int, help='number of examples')
args = parser.parse_args()


class SliceExample:
    def __init__(self, name, method, measurement):
        self.name = name
        self.method = method
        self.measurement = measurement

    def __repr__(self):
        return repr((self.name, self.method, self.measurement))


def check_args(methods, measures):
    flag, message = True, None
    if args.method not in methods:
        flag = False
        message = ' [!] Select method {} not in the consideration!'.format(args.method)

    if args.measure.lower() not in measures:
        flag = False
        message = ' [! Select measurement {} not in the consideration!'.format(args.measure)

    if args.number <= 0:
        flag = False
        message = ' [!] Number {} is small than 0!'.format(args.number)

    return flag, message


def read_imgs_and_measurement():
    slice_list = []

    # read gt image addresses
    gt_names = utils.all_files_under('../gt', extension='.jpg')

    for idx in range(len(gt_names)):
        if np.mod(idx, 300) == 0:
            print('Method: {}, Measure: {}, idx: {}'.format(args.method, args.measure, idx))

        img_name = os.path.basename(gt_names[idx])

        # read gt and prediction image
        gt_img = cv2.imread(gt_names[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        pred_img = cv2.imread(os.path.join('../{}'.format(args.method), img_name),
                              cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # calculate measurement
        measure_value = 0.
        if args.measure.lower() == 'mae':
            measure_value = utils.mean_absoulute_error(pred_img, gt_img)
        elif args.measure.lower() == 'rmse':
            measure_value = utils.root_mean_square_error(pred_img, gt_img)
        elif args.measure.lower() == 'psnr':
            measure_value = utils.peak_signal_to_noise_ratio(pred_img, gt_img)
        elif args.measure.lower() == 'ssim':
            measure_value = utils.structural_similarity_index(pred_img, gt_img)

        slice_list.append(SliceExample(img_name, args.method, measure_value))

    return slice_list


def main(methods, measures, img_size):
    flag, message = check_args(methods, measures)
    if not flag:
        sys.exit('{}'.format(message))

    save_folder = os.path.join('best{}_{}_{}'.format(args.number, args.measure, args.method))
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    if args.measure.lower() == 'mae' or args.measure.lower() == 'rmse':
        revise_flag = False
    else:  # 'psnr' and 'ssim'
        revise_flag = True

    slice_list = read_imgs_and_measurement()
    new_slice_list = sorted(slice_list, key=lambda slice_: slice_.measurement, reverse=revise_flag)

    # Save image
    for idx in range(args.number):
        font_type = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1.
        font_thickness = 2
        font_color = (255, 255, 255)

        # Calculate height and width of the text information
        info_str = '{} {:.3f}'.format(args.measure.upper(), new_slice_list[idx].measurement)
        info_size = cv2.getTextSize(info_str, font_type, font_scale, font_thickness)
        info_height, info_width = info_size[0][1], info_size[0][0]
        margin = int(0.2 * info_height)
        bottom_left = (int((len(methods) + 2) * img_size[1] * 0.5 - info_width * 0.5),
                       int(img_size[0] + info_height + 0.5 * margin))  # row and col

        img_name = new_slice_list[idx].name
        canvas = np.zeros((img_size[0] + info_height + margin, (len(methods) + 2) * img_size[1], 3), dtype=np.uint8)

        ct_img = cv2.imread(os.path.join('../ct', img_name))  # read 3 channel data
        mri_img = cv2.imread(os.path.join('../gt', img_name))  # read 3 channel data

        canvas[:-info_height-margin, :img_size[1], :] = ct_img
        canvas[:-info_height-margin, (len(methods)+1)*img_size[1]:, :] = mri_img

        for idx_method, method_name in enumerate(methods):
            img = cv2.imread(os.path.join('../{}'.format(method_name), img_name))  # read 3 channel data
            canvas[:-info_height-margin, (idx_method+1)*img_size[1]:(idx_method+2)*img_size[1], :] = img

        # Add text information
        cv2.putText(canvas, info_str, bottom_left,  font_type, font_scale, font_color, thickness=font_thickness)
        cv2.imwrite(os.path.join(save_folder, img_name), canvas)


if __name__ == '__main__':
    target_methods = ['pix2pix', 'cyclegan', 'discogan', 'mrgan']
    target_measures = ['mae', 'rmse', 'psnr', 'ssim']
    img_size_ = (300, 200, 1)

    main(target_methods, target_measures, img_size_)
