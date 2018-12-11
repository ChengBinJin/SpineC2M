import numpy as np
import matplotlib.pyplot as plt

n_groups = 5

means_men = (20, 35, 30, 35, 27)
std_men = (2, 3, 4, 1, 2)

means_women = (25, 32, 34, 20, 25)
std_women = (3, 5, 2, 3, 3)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.barh(index, means_men, bar_width, alpha=opacity, color='b', xerr=std_men, error_kw=error_config,
                label='pix2pix')

rects2 = ax.barh(index + bar_width, means_women, bar_width, alpha=opacity, color='r', xerr=std_women,
                error_kw=error_config, label='cyclegan')

rects3 = ax.barh(index + 2 * bar_width, means_women, bar_width, alpha=opacity, color='g', xerr=std_women,
                error_kw=error_config, label='discogan')

ax.set_xlabel('Scores')
ax.set_ylabel('Group')
ax.set_title('Scores by group and gender')
ax.set_yticks(index + bar_width)
ax.set_yticklabels(('Case 01', 'Case 02', 'Case 03', 'Case 04', 'Case 05'))
ax.invert_yaxis()  # labels read top-to-bottom
ax.legend()

fig.tight_layout()
plt.show()
