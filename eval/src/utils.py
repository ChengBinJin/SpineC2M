# ---------------------------------------------------------
# Python Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import cv2
import xlsxwriter
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
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
    colors = ['red', 'green', 'blue', 'aquamarine', 'aqua']  # purple

    for idx, data in enumerate(data_list):
        fig1, ax1 = plt.subplots(figsize=(2.5*len(method_names), 6))
        box = ax1.boxplot(np.transpose(data), patch_artist=True, showmeans=True, sym='r+', vert=True)

        # connect mean values
        y = data.mean(axis=1)
        ax1.plot(range(1, len(method_names)+1), y, 'r--')

        for patch, color in zip(box['boxes'], colors):
            patch.set(facecolor=color, alpha=0.5, linewidth=1)

        # scatter draw datapoints
        x_vals, y_vals = [], []
        for i in range(data_list[0].shape[0]):
            # move x coordinate to not overlapping
            x_vals.append(np.random.normal(i + 0.7, 0.04, data.shape[1]))
            y_vals.append(data[i, :].tolist())

        for x_val, y_val, color in zip(x_vals, y_vals, colors):
            ax1.scatter(x_val, y_val, s=5, c=color, alpha=0.5)

        ax1.yaxis.grid()  # horizontal lines
        ax1.set_xticklabels([method_name for method_name in method_names], fontsize=14)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        plt.setp(box['medians'], color='black')
        plt.title(filenames[idx] + expressions[idx], fontsize=14)
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

def difference_map(pred, gt, save_fold, img_name, measure, min_value=0, max_value=500):
    save_fold = os.path.join(save_fold, measure)
    if not os.path.isdir(save_fold):
        os.makedirs(save_fold)

    pred = (pred[:, :, 1]).astype(np.float32)
    gt = (gt[:, :, 1]).astype(np.float32)

    diff_map = np.abs(pred - gt)
    diff_map[0, 0] = min_value
    diff_map[-1, -1] = max_value

    plt.imshow(diff_map, vmin=min_value, vmax=max_value, cmap='afmhot')
    cb = plt.colorbar(ticks=np.linspace(min_value, max_value, num=3))
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=12)
    plt.axis('off')
    # plt.show()

    plt.savefig(os.path.join(save_fold, img_name), bbox_inches='tight')
    plt.close()


def central_crop(img, crop_size=(100, 80), start_h=110, start_w=60, factor=1.2, color=(0, 0, 255), thickness=1, base=1):
    # Crop image
    crop_img = img[start_h:start_h + crop_size[0], start_w:start_w + crop_size[1], :].copy()

    # Draw box on original image
    cv2.rectangle(img,
                  (start_w, start_h),
                  (start_w + crop_size[1], start_h + crop_size[0]),
                  color,
                  thickness)

    # Draw box on the croped image
    cv2.rectangle(crop_img,
                  (0 + base, 0 + base),
                  (crop_img.shape[1] - base, crop_img.shape[0] - base),
                  color,
                  thickness)

    # Past cropped image on the original one
    # img[:crop_size[0], -crop_size[1]:, :] = crop_img
    crop_img = cv2.resize(crop_img, (int(factor * crop_size[1]), int(factor * crop_size[0])),
                          interpolation=cv2.INTER_CUBIC)

    # Return cropped image
    return img, crop_img
