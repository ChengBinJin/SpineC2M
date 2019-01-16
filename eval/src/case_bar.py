# ---------------------------------------------------------
# Tensorflow SpineC2M Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import utils as utils


def count_cases(filenames):
    case_dict = {}

    for idx in range(len(filenames)):
        img_name = os.path.basename(filenames[idx])
        case_name = img_name[6:11]

        if case_name not in case_dict.keys():
            case_dict[case_name] = [img_name]
        else:
            case_dict[case_name].append(img_name)
            # case_list.append(case_name)

    return len(case_dict.keys()), case_dict


def cal_meausre(methods, measure, case_dict, num_cases_require):
    mean_bar = np.zeros((len(methods), num_cases_require), dtype=np.float32)
    var_bar = np.zeros((len(methods), num_cases_require), dtype=np.float32)

    case_idx = 0
    for key_idx in sorted(case_dict.keys()):
        if case_idx < num_cases_require:
            print('key_idx: {}'.format(key_idx))
            img_paths = case_dict[key_idx]

            measure_result = []
            for method in methods:
                method_measure = []
                for _, img_path in enumerate(img_paths):
                    gt_img = cv2.imread(os.path.join('../gt', img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    pred_img = cv2.imread(
                        os.path.join('../{}'.format(method), img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)

                    if measure.lower() == 'ssim':
                        measure_val = utils.structural_similarity_index(pred_img, gt_img)
                    else:
                        raise NotImplementedError

                    # Save one slice results
                    method_measure.append(measure_val)

                # Save whole slice results
                measure_result.append(method_measure)

            measure_array = np.asarray(measure_result)
            mean_bar[:, case_idx] = np.mean(measure_array, axis=1)
            var_bar[:, case_idx] = np.std(measure_array, axis=1)

            case_idx += 1

        else:
            break

    return mean_bar, var_bar


def horizontal_bar_plot(methods, mean_arrs, var_arrs, num_cases_require):
    index = np.arange(num_cases_require)
    bar_width = 0.2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    colors = ['red', 'green', 'blue', 'yellow']  # ['r', 'g', 'b']

    fig, ax = plt.subplots(figsize=(5*len(methods), 10))
    # Add three meathods bar plot
    for idx in range(len(methods)):
        ax.bar(index + idx * bar_width, mean_arrs[idx], bar_width, alpha=opacity, color=colors[idx],
                yerr=var_arrs[idx], error_kw=error_config, label=methods[idx])

    ax.set_ylabel('SSIM', fontsize=16)
    ax.set_xlabel('Subject', fontsize=16)
    ax.set_xticks(index + bar_width * len(methods) / 2 - bar_width / 2)
    ax.set_xticklabels([str(case_id).zfill(2) for case_id in range(1, num_cases_require+1)], fontsize=14)
    ax.set_yticks(np.arange(0., 0.5, 0.05))
    ax.set_yticklabels(['{:.2f}'.format(ytick) for ytick in np.arange(0., 0.5, 0.05)], fontsize=14)
    ax.legend(fontsize=18)

    fig.tight_layout()
    plt.savefig('CaseBarPlot.jpg', dpi=600)
    plt.close()

def main(methods, display_names, measure, num_cases_require):
    # Read gt image paths
    gt_names = utils.all_files_under('../gt', extension='.jpg')
    num_cases, case_dict = count_cases(gt_names)  # sort and save img paths according to subject id

    # Calculate ssim according to case id
    mean_arrs, var_arrs = cal_meausre(methods, measure, case_dict, num_cases_require)

    # Horizontal bar plot
    horizontal_bar_plot(display_names, mean_arrs, var_arrs, num_cases_require)


if __name__ == '__main__':
    target_methods = ['pix2pix', 'cyclegan', 'discogan', 'mrgan']
    display_names_ = ['Multi-Channel GAN', 'Deep MR-to-CT', 'DiscoGAN', 'MR-GAN']
    target_measusre = 'ssim'
    num_cases_require_ = 20

    main(target_methods, display_names_, target_measusre, num_cases_require_)
