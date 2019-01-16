# ---------------------------------------------------------
# Tensorflow SpineC2M Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import cv2
import xlsxwriter
import numpy as np
import argparse

import utils as utils

parser = argparse.ArgumentParser(description='Hello parser')
parser.add_argument('--method', dest='method', default='mrgan', help='separate method name')
args = parser.parse_args()


def write_to_csv(save_name, data_list, img_names):
    num_tests = data_list[0].shape[0]
    print('num_tests: {}'.format(num_tests))

    # Create a workbook and add a worksheet
    xlsx_name = 'data/{}.xlsx'.format(save_name)
    workbook = xlsxwriter.Workbook(xlsx_name)
    xlsFormate = workbook.add_format()
    xlsFormate.set_align('center')
    xlsFormate.set_valign('vcenter')

    # calucate mean of the MAE and PSNR for one method
    mean_mae = np.mean(data_list[0])
    mean_mse = np.mean(data_list[1])
    mean_psnr = np.mean(data_list[2])
    mean_ssim = np.mean(data_list[3])
    mean_pcc = np.mean(data_list[4])
    std_mae = np.std(data_list[0])
    std_mse = np.std(data_list[1])
    std_psnr = np.std(data_list[2])
    std_ssim = np.std(data_list[3])
    std_pcc = np.std(data_list[4])

    attributes = ['No', 'Name', 'MAE', 'RMSE', 'PSNR', 'SSIM', 'PCC']
    worksheet = workbook.add_worksheet(name=save_name)
    for attr_idx in range(len(attributes)):
        worksheet.write(0, attr_idx, attributes[attr_idx], xlsFormate)

    for img_idx in range(num_tests):
        for attr_idx in range(len(attributes)):
            if attr_idx == 0:
                worksheet.write(img_idx+1, attr_idx, str(img_idx).zfill(4), xlsFormate)
            elif attr_idx == 1:
                worksheet.write(img_idx+1, attr_idx, img_names[img_idx][6:], xlsFormate)
            else:
                worksheet.write(img_idx+1, attr_idx, data_list[attr_idx-2][img_idx], xlsFormate)

        # write mean and std value for MAE, MSE, and PSNR
        worksheet.write(num_tests+1, 1, 'Mean', xlsFormate)
        worksheet.write(num_tests+1, 2, mean_mae, xlsFormate)
        worksheet.write(num_tests+1, 3, mean_mse, xlsFormate)
        worksheet.write(num_tests+1, 4, mean_psnr, xlsFormate)
        worksheet.write(num_tests+1, 5, mean_ssim, xlsFormate)
        worksheet.write(num_tests+1, 6, mean_pcc, xlsFormate)

        worksheet.write(num_tests+2, 1, 'Std', xlsFormate)
        worksheet.write(num_tests+2, 2, std_mae, xlsFormate)
        worksheet.write(num_tests+2, 3, std_mse, xlsFormate)
        worksheet.write(num_tests+2, 4, std_psnr, xlsFormate)
        worksheet.write(num_tests+2, 5, std_ssim, xlsFormate)
        worksheet.write(num_tests+2, 6, std_pcc, xlsFormate)

    workbook.close()


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

    data_list = [mae_method, rmse_method, psnr_method, ssim_method, pcc_method]
    write_to_csv(method, data_list, gt_names)

if __name__ == '__main__':
    gt_ = 'gt'
    main(gt_, args.method)
