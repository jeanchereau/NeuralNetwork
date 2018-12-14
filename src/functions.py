import numpy as np
import matplotlib.pyplot as plt


# Function to retrieve k nearest neighbours with euclidean distance (l2 norm).
def knn(x_s, x_data, k=1):
    x_dist = np.linalg.norm(x_data - x_s[None, :], axis=1)

    idx = np.argsort(x_dist)[0:k]

    return idx


# Returns real parts of eigenvectors associated with m largest eigenvalues in decreasing order.
def eigen_order(matrix, m=None):
    lc, vc = np.linalg.eig(matrix)

    indices = np.argsort(lc.real)[::-1]

    plt.figure()
    plt.plot(lc[indices].real)
    plt.title('Eigenvalues of Principle Components')

    v = vc[:, indices]
    v = v[:, 0:m]

    return v.real


# Display retrieval results for given query
def result_display(rank, color, image_files):
    color_list = ['black', 'green', 'red']

    fig = plt.figure(figsize=(rank + 3, 3))
    cols, rows = rank + 1, 1

    i = 0
    for image_file in image_files:
        image = plt.imread('../pr_data/images_cuhk03/' + image_file[0])
        ax = fig.add_subplot(rows, cols, i+1)
        plt.imshow(image, aspect='auto')
        plt.setp(ax.spines.values(), color=color_list[color[i]], linewidth=3)
        ax.tick_params(bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        i += 1

    plt.show()
