import numpy as np
import matplotlib.pyplot as plt


def knn(x_s, x_data, k=1):
    x_dist = np.linalg.norm(x_data - x_s[None, :], axis=1)

    idx = np.argsort(x_dist)[0:k]

    return idx


# Returns indices of k features closest to input feature according to given metric.
def kmetric(x_s, x_data, quad_mat, k=1):

    x_dist = np.sqrt(np.diag((x_data - x_s[None, :]).dot(quad_mat.dot((x_data - x_s[None, :]).T))))

    idx = np.argsort(x_dist)[0:k]

    return idx


def eigen_order(matrix, m=None):
    lc, vc = np.linalg.eig(matrix)

    indices = np.argsort(lc.real)[::-1]

    plt.figure()
    plt.plot(lc[indices].real)
    plt.title('Eigenvalues of principle components.')

    v = vc[:, indices]
    v = v[:, 0:m]

    return v.real


def result_display(rank, color, image_files):
    color_dict = ['black', 'green', 'red']
    w = rank + 3
    h = 3
    fig = plt.figure(figsize=(w, h))
    cols = rank + 1
    rows = 1

    i = 0
    for image_file in image_files:
        image = plt.imread('../pr_data/images_cuhk03/' + image_file[0])
        ax = fig.add_subplot(rows, cols, i+1)
        plt.imshow(image, aspect='auto')
        plt.setp(ax.spines.values(), color=color_dict[color[i]], linewidth=3)
        ax.tick_params(bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        i += 1

    plt.show()


def f_measure(prec, recall):

    db = 0.01
    beta = np.arange(db, 100, db)

    f = (1 + np.square(beta)) * (prec * recall) / (np.square(beta) * prec + recall)

    fig, ax = plt.subplots()

    ax.semilogx(beta, f)
    ax.grid()

    plt.show()
