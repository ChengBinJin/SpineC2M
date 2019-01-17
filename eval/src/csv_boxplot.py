# ---------------------------------------------------------
# Tensorflow SpineC2M Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import xlrd
import numpy as np

import utils as utils

def main(methods, display_names, num_tests=4426):
    # measures = ['MAE', 'RMSE', 'PSNR', 'SSIM', 'PCC']
    mae_overall = np.zeros((len(methods), num_tests), dtype=np.float64)
    rmse_overall = np.zeros((len(methods), num_tests), dtype=np.float64)
    psnr_overall = np.zeros((len(methods), num_tests), dtype=np.float64)
    ssim_overall = np.zeros((len(methods), num_tests), dtype=np.float64)
    pcc_overall = np.zeros((len(methods), num_tests), dtype=np.float64)

    for method_idx, method in enumerate(methods):
        print('method_idx: {}'.format(method_idx))
        workbook = xlrd.open_workbook(os.path.join('data', method+'.xlsx'))
        worksheet = workbook.sheet_by_name(method)

        for row_idx in range(1, worksheet.nrows-2):
            mae_overall[method_idx, row_idx-1] = float(worksheet.cell(row_idx, 2).value)
            rmse_overall[method_idx, row_idx-1] = float(worksheet.cell(row_idx, 3).value)
            psnr_overall[method_idx, row_idx-1] = float(worksheet.cell(row_idx, 4).value)
            ssim_overall[method_idx, row_idx-1] = float(worksheet.cell(row_idx, 5).value)
            pcc_overall[method_idx, row_idx-1] = float(worksheet.cell(row_idx, 6).value)

    # draw boxplot
    utils.draw_box_plot([mae_overall, rmse_overall, psnr_overall, ssim_overall, pcc_overall], display_names)


if __name__ == '__main__':
    methods_ = ['pix2pix', 'cyclegan', 'discogan', 'mrgan', 'mrganPlus_w1=100_w2=1']
    display_names_ = ['Multi-Channel GAN', 'Deep MR-to-CT', 'DiscoGAN', 'MR-GAN', 'MR-GAN+']

    main(methods_, display_names_)
