import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rank_display(rank, color, *image_files):
    n_images = rank + 1
    color_dict = {'black': 0, 'green': 1, 'red': 2}

    plt.figure()

    for image_file in image_files:
        image = mpimg.imread(image_file)

        imgplot = plt.imshow(image)

    plt.show()
