# ---------------------------------------------------------
# Python Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import xlsxwriter
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from scipy.stats import pearsonr


def write_to_csv(data_list, method_names, img_names):
    num_tests = data_list[0].shape[1]

    # Create a workbook and add a worksheet
    xlsx_name = 'statistics.xlsx'
    workbook = xlsxwriter.Workbook(xlsx_name)
    xlsFormate = workbook.add_format()
    xlsFormate.set_align('center')
    xlsFormate.set_valign('vcenter')

    # calucate mean of the MAE and PSNR for one method
    mean_mae = np.mean(data_list[0], axis=1)
    mean_mse = np.mean(data_list[1], axis=1)
    mean_psnr = np.mean(data_list[2], axis=1)
    mean_ssim = np.mean(data_list[3], axis=1)
    mean_pcc = np.mean(data_list[4], axis=1)
    std_mae = np.std(data_list[0], axis=1)
    std_mse = np.std(data_list[1], axis=1)
    std_psnr = np.std(data_list[2], axis=1)
    std_ssim = np.std(data_list[3], axis=1)
    std_pcc = np.std(data_list[4], axis=1)

    attributes = ['No', 'Name', 'MAE', 'RMSE', 'PSNR', 'SSIM', 'PCC']
    for idx in range(len(method_names)):
        worksheet = workbook.add_worksheet(name=method_names[idx])
        for attr_idx in range(len(attributes)):
            worksheet.write(0, attr_idx, attributes[attr_idx], xlsFormate)

        for img_idx in range(num_tests):
            for attr_idx in range(len(attributes)):
                if attr_idx == 0:
                    worksheet.write(img_idx+1, attr_idx, str(img_idx).zfill(4), xlsFormate)
                elif attr_idx == 1:
                    worksheet.write(img_idx+1, attr_idx, img_names[img_idx][6:], xlsFormate)
                else:
                    worksheet.write(img_idx+1, attr_idx, data_list[attr_idx-2][idx, img_idx], xlsFormate)

        # write mean and std value for MAE, MSE, and PSNR
        worksheet.write(num_tests+1, 1, 'Mean', xlsFormate)
        worksheet.write(num_tests+1, 2, mean_mae[idx], xlsFormate)
        worksheet.write(num_tests+1, 3, mean_mse[idx], xlsFormate)
        worksheet.write(num_tests+1, 4, mean_psnr[idx], xlsFormate)
        worksheet.write(num_tests+1, 5, mean_ssim[idx], xlsFormate)
        worksheet.write(num_tests+1, 6, mean_pcc[idx], xlsFormate)

        worksheet.write(num_tests+2, 1, 'Std', xlsFormate)
        worksheet.write(num_tests+2, 2, std_mae[idx], xlsFormate)
        worksheet.write(num_tests+2, 3, std_mse[idx], xlsFormate)
        worksheet.write(num_tests+2, 4, std_psnr[idx], xlsFormate)
        worksheet.write(num_tests+2, 5, std_ssim[idx], xlsFormate)
        worksheet.write(num_tests+2, 6, std_pcc[idx], xlsFormate)

    workbook.close()


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def draw_box_plot(data_list, method_names):
    filenames = ['MAE', 'RMSE', 'PSNR', 'SSIM', 'PCC']
    expressions = [' (lower is better)', ' (lower is better)', ' (higher is better)', ' (higher is better)',
                   '(higher is better)']
    colors = ['red', 'green', 'blue']  # ['blue', 'red', 'green', 'yellow']

    for idx, data in enumerate(data_list):
        box = plt.boxplot(np.transpose(data), patch_artist=True, showmeans=True, sym='r+', vert=True)

        # connect mean values
        y = data.mean(axis=1)
        plt.plot(range(1, len(method_names)+1), y, 'r--')

        for patch, color in zip(box['boxes'], colors):
            patch.set(facecolor=color, alpha=0.5, linewidth=1)

        # scatter draw datapoints
        x_vals, y_vals = [], []
        for i in range(data_list[0].shape[0]):
            # move x coordinate to not overlapping
            x_vals.append(np.random.normal(i + 0.7, 0.04, data.shape[1]))
            y_vals.append(data[i, :].tolist())

        for x_val, y_val, color in zip(x_vals, y_vals, colors):
            plt.scatter(x_val, y_val, s=5, c=color, alpha=0.5)

        ax = plt.axes()
        ax.yaxis.grid()  # horizontal lines
        plt.xticks(range(1, len(method_names)+1), method_names)
        plt.setp(box['medians'], color='black')
        plt.title(filenames[idx] + expressions[idx])
        plt.savefig(filenames[idx] + '.jpg', dpi=300)
        plt.close()


def mean_absoulute_error(pred, gt):
    h, w = pred.shape
    mae = np.sum(np.abs(pred - gt)) / (h * w)
    return mae


def root_mean_square_error(pred, gt):
    h, w = pred.shape
    rmse = np.sqrt(np.sum(np.square(pred - gt)) / (h * w))
    return rmse


def peak_signal_to_noise_ratio(pred, gt):
    max_value = 255. * 255.
    h, w = pred.shape
    upper_bound = 20 * np.log10(max_value)
    psnr = upper_bound - 10 * np.log10(np.sum(np.square(pred - gt)) / (h * w))
    return psnr


def structural_similarity_index(pred, gt):
    # Use skimage.measure li
    return compare_ssim(gt, pred, data_range=pred.max() - pred.min())


def pearson_correlation_coefficient(pred, gt):
    coeff, _ = pearsonr(gt.ravel(), pred.ravel())
    return coeff
