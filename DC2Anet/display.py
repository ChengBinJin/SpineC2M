import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def plot_statistics(data, id_names):
    x = range(len(data))  # the label locations
    labels = id_names
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x, data, width, align='center', alpha=0.5)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Num of slices')
    ax.set_title('Statistics of number of slices for each object')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.xticks(rotation=60)  # rotate xticks in 60 degree

    fig.tight_layout()
    plt.show()


def main():
    data_path = '../../Data/brain07/'
    normal_folders = [os.path.join(data_path, 'normal', fname) for fname in os.listdir(data_path + 'normal')]
    tumor_folders = [os.path.join(data_path, 'tumor', fname) for fname in os.listdir(data_path + 'tumor')]

    num_normal_cases = len(normal_folders)
    num_tumor_cases = len(tumor_folders)

    normal_ct_slices = normal_mr_slices = 0
    tumor_ct_slices = tumor_mr_slices = 0

    statics_slices = list()  # save number of slices for each object
    id_names = list()

    for normal_folder in normal_folders:
        id_names.append(os.path.basename(normal_folder))

        for i, sub_folder in enumerate(['CT', 'warped']):
            folder_name = os.path.join(normal_folder, sub_folder)
            data_path = [os.path.join(folder_name, fname) for fname in os.listdir(folder_name) if fname.endswith('.nii')]

            # if there are two files in CT, read the first one
            data = nib.load(data_path[0]).get_fdata()

            if i == 0:
                normal_ct_slices += data.shape[-1]
                statics_slices.append(data.shape[-1])
            else:
                normal_mr_slices += data.shape[-1]

    for tumor_folder in tumor_folders:
        id_names.append(os.path.basename(tumor_folder))

        for i, sub_folder in enumerate(['CT', 'warped']):
            folder_name = os.path.join(tumor_folder, sub_folder)
            data_path = [os.path.join(folder_name, fname) for fname in os.listdir(folder_name) if
                         fname.endswith('.nii')]

            # if there are two files in CT, read the first one
            data = nib.load(data_path[0]).get_fdata()

            if i == 0:
                tumor_ct_slices += data.shape[-1]
                statics_slices.append(data.shape[-1])
            else:
                tumor_mr_slices += data.shape[-1]

    plot_statistics(statics_slices, id_names)  # plot number of slices for each object

    print('Num of normal CT slices: {}'.format(normal_ct_slices))
    print('Num of normal MR slices: {}'.format(normal_mr_slices))
    print('Num of tumor CT slices: {}'.format(tumor_ct_slices))
    print('Num of tumor MR slices: {}'.format(tumor_mr_slices))
    print('Num. of normal cases: {}'.format(num_normal_cases))
    print('Num. of tumor cases: {}'.format(num_tumor_cases))


if __name__ == '__main__':
    main()



