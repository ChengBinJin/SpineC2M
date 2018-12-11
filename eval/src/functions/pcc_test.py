import cv2
import numpy as np

from scipy.stats import pearsonr
x = [1, 2, 3, 4, 5]

corr = [2, 4, 6, 8, 10]
corr, p_value = pearsonr(x, corr)

print(corr)


img_gt = cv2.imread('../../gt/p0000_c0001_s0001_mri.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
img_cycle = cv2.imread('../../cyclegan/p0000_c0001_s0001_mri.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

corr, p_value = pearsonr(img_gt.ravel(), img_cycle.ravel())
print(corr)
