import cv2
import numpy as np
import matplotlib.pyplot as plt

def difference_map(pred, gt, min_value=0, max_value=500):
    pred = (pred[:, :, 1]).astype(np.float32)
    gt = (gt[:, :, 1]).astype(np.float32)

    diff_map = np.abs(pred - gt)
    diff_map[0, 0] = min_value
    diff_map[-1, -1] = max_value

    plt.imshow(diff_map, vmin=min_value, vmax=max_value, cmap='afmhot')
    cb = plt.colorbar(ticks=np.linspace(min_value, max_value, num=3))
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=12)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    gt_img = cv2.imread('../../gt/p0000_c0001_s0001_mri.jpg')
    pred_img = cv2.imread('../../mrganPlus/p0000_c0001_s0001_mri.jpg')

    difference_map(pred_img, gt_img)
