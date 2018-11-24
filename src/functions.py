import numpy as np
import matplotlib.pyplot as plt


def knn(x_s, x_data, k=1):
    cols = x_s.size

    x_dist = np.linalg.norm(x_data - x_s.reshape(1, cols), axis=1)

    idx = np.argsort(x_dist)[0:k]

    return idx


def rank_display(rank, color, *image_files):
    color_dict = {'black': 0, 'green': 1, 'red': 2}
    w = 10
    h = 5
    fig = plt.figure(figsize=(8, 8))
    cols = rank + 1
    rows = 1

    i = 1
    for image_file in image_files:
        image = plt.imread('../pr_data/images_cuhk03/' + image_file)
        fig.add_subplot(rows, cols, i)
        i += 1
        imgplot = plt.imshow(image)

    plt.show()

    # for ax, color in zip([ax1, ax2, ax3, ax4], ['green', 'green', 'blue', 'blue']):
    #    plt.setp(ax.spines.values(), color=color)
    #    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=color)


def rank_map():
    pass
