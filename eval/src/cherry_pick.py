import os
import sys
import cv2
import argparse
import numpy as np

import utils as utils

parser = argparse.ArgumentParser(description='')
parser.add_argument('--method', dest='method', default='pix2pix', type=str,
                    help='select from [pix2pix|cyclegan|discogan]')
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
            print('Method: {}, Meausre: {}, idx: {}'.format(args.method, args.measure, idx))

        img_name = os.path.basename(gt_names[idx])

        # read gt and prediction image
        gt_img = cv2.imread(gt_names[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        pred_img = cv2.imread(os.path.join('../{}'.format(args.method), img_name),
                              cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # calcualte measurement
        measure_value = 0.
        if args.measure == 'mae':
            measure_value = utils.mean_absoulute_error(pred_img, gt_img)
        elif args.measure == 'rmse':
            measure_value = utils.root_mean_square_error(pred_img, gt_img)
        elif args.measure == 'psnr':
            measure_value = utils.peak_signal_to_noise_ratio(pred_img, gt_img)
        elif args.measure == 'ssim':
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

    slice_list = read_imgs_and_measurement()
    new_slice_list = sorted(slice_list, key=lambda slice_: slice_.measurement, reverse=True)

    # Save image
    for idx in range(args.number):
        img_name = new_slice_list[idx].name + '.jpg'
        canvas = np.zeros((img_size[0], (len(methods) + 2) * img_size[1]), dtype=np.uint8)

        ct_img = cv2.imread(os.path.join('../ct', img_name), cv2.IMREAD_GRAYSCALE)
        mri_img = cv2.imread(os.path.join('../gt', img_name), cv2.IMREAD_GRAYSCALE)
        print('img_size[1] {}'.format(img_size[1]))
        canvas[:, :img_size[1]] = ct_img
        canvas[:, (len(methods)+1)*img_size[1]:] = mri_img

        for idx_method, method_name in enumerate(methods):
            img = cv2.imread(os.path.join('../{}'.format(method_name), img_name), cv2.IMREAD_GRAYSCALE)
            canvas[: (idx_method+1)*img_size[1]:(idx_method+2)*img_size[1]] = img

        cv2.imwrite(os.path.join(save_folder, img_name), canvas)


if __name__ == '__main__':
    target_methods = ['pix2pix', 'cyclegan', 'discogan']
    target_measures = ['mae', 'rmse', 'psnr', 'ssim']
    img_size_ = (300, 200, 1)

    main(target_methods, target_measures, img_size_)
