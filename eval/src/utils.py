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
    std_mae = np.std(data_list[0], axis=1)
    std_mse = np.std(data_list[1], axis=1)
    std_psnr = np.std(data_list[2], axis=1)

    attributes = ['No', 'Name', 'MAE', 'RMSE', 'PSNR']
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

        worksheet.write(num_tests+2, 1, 'Std', xlsFormate)
        worksheet.write(num_tests+2, 2, std_mae[idx], xlsFormate)
        worksheet.write(num_tests+2, 3, std_mse[idx], xlsFormate)
        worksheet.write(num_tests+2, 4, std_psnr[idx], xlsFormate)

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
    filenames = ['MAE', 'RMSE', 'PSNR']
    expressions = [' (lower is better)', ' (lower is better)', ' (higher is better)']
    colors = ['red', 'green']  # ['blue', 'red', 'green', 'yellow']

    for idx, data in enumerate(data_list):
        box = plt.boxplot(np.transpose(data), patch_artist=True, showmeans=True, sym='r+')

        # connect mean values
        y = data.mean(axis=1)
        plt.plot([1, 2], y, 'r--')

        for patch, color in zip(box['boxes'], colors):
            patch.set(facecolor=color, alpha=0.5, linewidth=1)

        ax = plt.axes()
        ax.yaxis.grid()  # horizontal lines
        plt.xticks([1, 2], method_names)
        plt.setp(box['medians'], color='black')
        plt.title(filenames[idx] + expressions[idx])
        plt.savefig(filenames[idx] + '.jpg', dpi=300)
        plt.close()


def mean_absoulute_error(pred_, gt_):
    pred, gt = pred_.copy(), gt_.copy()
    h, w = pred.shape
    mae = np.sum(np.abs(pred - gt)) / (h * w)

    return mae


def root_mean_square_error(pred_, gt_):
    pred, gt = pred_.copy(), gt_.copy()
    h, w = pred.shape
    mse = np.sqrt(np.sum(np.square(pred - gt)) / (h * w))
    return mse


def peak_signal_to_noise_ratio(pred_, gt_):
    pred, gt = pred_.copy(), gt_.copy()
    max_value = 255. * 255.
    h, w = pred.shape
    upper_bound = 20 * np.log10(max_value)
    psnr = upper_bound - 10 * np.log10(np.sum(np.square(pred - gt)) / (h * w))
    return psnr
