# ---------------------------------------------------------
# Tensorflow SpineC2M Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by vanhuyz
# ---------------------------------------------------------
import os
import sys
import cv2
import numpy as np
import argparse

import utils as utils

parser = argparse.ArgumentParser(description='Hello parser')
parser.add_argument('--method', dest='method', default='mrgan', help='separate method name')
args = parser.parse_args()


def main(gt, method):
    # read gt image addresses
    gt_names = utils.all_files_under(os.path.join('../', gt), extension='.jpg')

    # read prediction image addresses
    filenames = utils.all_files_under(os.path.join('../', method), extension='.jpg')

    # print(methods[idx_method])
    mae_method, rmse_method, psnr_method, ssim_method, pcc_method = [], [], [], [], []

    for idx_name in range(len(gt_names)):
        # read gt and prediction image
        gt_img = cv2.imread(gt_names[idx_name], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        pred_img = cv2.imread(filenames[idx_name], cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # check gt and prediction image name
        if gt_names[idx_name].split('p0')[-1] != filenames[idx_name].split('p0')[-1]:
            sys.exit(' [!] Image name can not match!')

        # calcualte mae and psnr
        mae = utils.mean_absoulute_error(pred_img, gt_img)
        rmse = utils.root_mean_square_error(pred_img, gt_img)
        psnr = utils.peak_signal_to_noise_ratio(pred_img, gt_img)
        ssim = utils.structural_similarity_index(pred_img, gt_img)
        pcc = utils.pearson_correlation_coefficient(pred_img, gt_img)

        if np.mod(idx_name, 300) == 0:
            print('Method: {}, idx: {}'.format(method, idx_name))

        # collect each image results
        mae_method.append(mae)
        rmse_method.append(rmse)
        psnr_method.append(psnr)
        ssim_method.append(ssim)
        pcc_method.append(pcc)

    # list to np.array
    mae_method = np.asarray(mae_method)
    rmse_method = np.asarray(rmse_method)
    psnr_method = np.asarray(psnr_method)
    ssim_method = np.asarray(ssim_method)
    pcc_method = np.asarray(pcc_method)

    print(' MAE - mean: {:.3f}, std: {:.3f}'.format(np.mean(mae_method), np.std(mae_method)))
    print('RMSE - mean: {:.3f}, std: {:.3f}'.format(np.mean(rmse_method), np.std(rmse_method)))
    print('PSNR - mean: {:.3f}, std: {:.3f}'.format(np.mean(psnr_method), np.std(psnr_method)))
    print('SSIM - mean: {:.3f}, std: {:.3f}'.format(np.mean(ssim_method), np.std(ssim_method)))
    print(' PCC - mean: {:.3f}, std: {:.3f}'.format(np.mean(pcc_method), np.std(pcc_method)))

if __name__ == '__main__':
    gt_ = 'gt'
    main(gt_, args.method)
