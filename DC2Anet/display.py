import cv2
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def plot_statistics(data, id_names):
    x = range(len(data))  # the label locations
    labels = id_names
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.bar(x, data, width, align='center', alpha=0.5)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of slices', fontsize=14)
    ax.set_xlabel('Patient ID', fontsize=14)
    ax.set_title('Statistics of Number of Slices for Each Patient', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.grid()
    plt.xticks(rotation=60)  # rotate xticks in 60 degree

    fig.tight_layout()
    plt.savefig('./img/statistics_of_number_of_slices_for_each_patient.png', dpi=600)
    plt.show()


def save_img_slice(ct_data, mr_data, subfolder, filename):
    height, width, depth = ct_data.shape
    for i in range(ct_data.shape[-1]):
        canvas = np.zeros((height, 2 * width), dtype=np.uint8)
        ct_slice = ct_data[:, :, i].copy()
        mr_slice = mr_data[:, :, i].copy()

        canvas[:, :width] = (ct_slice / ct_slice.max() * 255).astype(np.uint8)
        canvas[:, width:] = (mr_slice / mr_slice.max() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join('./img', subfolder, filename + '_' + str(i).zfill(3) + '.png'), canvas)


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

    # Normal data
    for i, folders in enumerate([normal_folders, tumor_folders]):
        for folder in folders:
            file_name = os.path.basename(folder)
            id_names.append(os.path.basename(folder))
            print('ID name: {}'.format(file_name))

            ct_folder_name = os.path.join(folder, 'CT')
            mr_folder_name = os.path.join(folder, 'warped')

            ct_data_path = [os.path.join(ct_folder_name, fname)
                            for fname in os.listdir(ct_folder_name) if fname.endswith('.nii')]
            mr_data_path = [os.path.join(mr_folder_name, fname)
                            for fname in os.listdir(mr_folder_name) if fname.endswith('.nii')]

            # if there are two files in CT, read the first one
            ct_data = nib.load(ct_data_path[0]).get_fdata()
            mr_data = nib.load(mr_data_path[0]).get_fdata()

            # Save ct & mr slice as alinged data
            save_img_slice(ct_data, mr_data, subfolder='normal' if i == 0 else 'tumor', filename=file_name)

            # Count total number of slices for each object
            if i == 0:
                normal_ct_slices += ct_data.shape[-1]
                normal_mr_slices += mr_data.shape[-1]
            elif i == 1:
                tumor_ct_slices += ct_data.shape[-1]
                tumor_mr_slices += mr_data.shape[-1]

            statics_slices.append(ct_data.shape[-1])

    plot_statistics(statics_slices, id_names)  # plot number of slices for each object

    print('Num of normal CT slices: {}'.format(normal_ct_slices))
    print('Num of normal MR slices: {}'.format(normal_mr_slices))
    print('Num of tumor CT slices: {}'.format(tumor_ct_slices))
    print('Num of tumor MR slices: {}'.format(tumor_mr_slices))
    print('Num. of normal cases: {}'.format(num_normal_cases))
    print('Num. of tumor cases: {}'.format(num_tumor_cases))


if __name__ == '__main__':
    main()



