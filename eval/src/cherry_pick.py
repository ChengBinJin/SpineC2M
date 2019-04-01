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
parser.add_argument('--method', dest='method', default='DC2Anet', type=str,
                    help='select from [pix2pix|cyclegan|discogan|DC2Anet]')
parser.add_argument('--measure', dest='measure', default='ssim', type=str,
                    help='select from [mae|rmse|psnr|ssim]')
parser.add_argument('--number', dest='number', default=20, type=int,
                    help='number of examples')
parser.add_argument('--add_value', dest='add_value', default=False, action='store_true',
                    help='add value in image or not')
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

    save_folder = 'best{}_{}_{}'.format(args.number, args.measure, args.method)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    if args.measure.lower() == 'mae' or args.measure.lower() == 'rmse':
        revise_flag = False
    else:  # 'psnr' and 'ssim'
        revise_flag = True

    slice_list = read_imgs_and_measurement()
    new_slice_list = sorted(slice_list, key=lambda slice_: slice_.measurement, reverse=revise_flag)

    save_imgs(new_slice_list, save_folder, methods, img_size)

def save_imgs(slice_list, save_folder, methods, img_size, crop_size=(100, 80), factor=1.5):
    # Save image
    for idx in range(args.number):
        canvas, info_height, margin = None, None, None

        if args.add_value:
            font_type = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 1.
            font_thickness = 2
            font_color = (0, 0, 0)

            # Calculate height and width of the text information
            info_str = '{} {:.3f}'.format(args.measure.upper(), slice_list[idx].measurement)
            info_size = cv2.getTextSize(info_str, font_type, font_scale, font_thickness)
            info_height, info_width = info_size[0][1], info_size[0][0]
            margin = int(0.2 * info_height)
            bottom_left = (int((len(methods) + 2) * img_size[1] * 0.5 - info_width * 0.5),
                           int(img_size[0] + int(factor * crop_size[0]) + info_height + 0.5 * margin))  # row and col

            canvas = 255 * np.ones((img_size[0] + int(factor * crop_size[0]) + info_height + margin,
                                    (len(methods) + 2) * img_size[1], 3), dtype=np.uint8)

            # Add text information
            cv2.putText(canvas, info_str, bottom_left, font_type, font_scale, font_color, thickness=font_thickness)
        else:
            canvas = 255 * np.ones((img_size[0] + int(factor * crop_size[0]), (len(methods) + 2) * img_size[1], 3),
                                   dtype=np.uint8)

        img_name = slice_list[idx].name
        ct_img = cv2.imread(os.path.join('../ct', img_name))  # read 3 channel data
        mri_img = cv2.imread(os.path.join('../gt', img_name))  # read 3 channel data

        canvas[:img_size[0], :img_size[1], :] = ct_img  # save ct img
        # save mri img
        mri_img_rec, mri_crop = utils.central_crop(mri_img.copy(), factor=factor)
        canvas[:img_size[0], (len(methods)+1)*img_size[1]:, :] = mri_img_rec
        canvas[img_size[0]:img_size[0] + int(factor * crop_size[0]), -int(factor * crop_size[1]):, :] = mri_crop

        for idx_method, method_name in enumerate(methods):
            # Read 3 channel data
            img = cv2.imread(os.path.join('../{}'.format(method_name), img_name))
            # calculate difference map
            utils.difference_map(img, mri_img, save_folder, img_name, method_name)

            img, crop_img = utils.central_crop(img, factor=factor)
            if args.add_value:
                canvas[:-int(factor * crop_size[0]) - info_height - margin, (idx_method + 1) * img_size[1]:(idx_method + 2) * img_size[1], :] = img
                canvas[img_size[0]:-info_height-margin, (idx_method + 2) * img_size[1] - int(factor * crop_size[1]):(idx_method + 2) * img_size[1], :] = crop_img
            else:
                canvas[:img_size[0], (idx_method+1)*img_size[1]:(idx_method+2)*img_size[1], :] = img
                canvas[img_size[0]:, (idx_method+2)*img_size[1]-int(factor*crop_size[1]):(idx_method+2)*img_size[1], :] = crop_img

        cv2.imwrite(os.path.join(save_folder, img_name), canvas)


if __name__ == '__main__':
    target_methods = ['pix2pix', 'cyclegan', 'discogan', 'mrgan', 'DC2Anet']
    target_measures = ['mae', 'rmse', 'psnr', 'ssim', 'pcc']
    img_size_ = (300, 200, 1)

    main(target_methods, target_measures, img_size_)
