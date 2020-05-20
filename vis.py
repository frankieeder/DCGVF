import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def show_sample(m):
    plt.imshow(np.transpose(m, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def img_hist(im):
    print(im)
    print(im.shape)
    print(im.dtype)
    a = plt.hist(im.flatten(), bins=2**10, range=(0, 1), density=True)
    #plt.ylim(0, a[0].max())
    plt.show()

def visual_results(s, name):
    x = s['x'][0]
    y = s['y'][0]
    y_hat = s['y_hat']#[0]

    diff = x[3] - y
    diff_vals = diff.flatten()
    diff -= diff.min()
    max_diff = diff.max()
    diff /= max_diff

    res = y_hat - y
    res_vals = res.flatten()
    res -= res.min()
    res /= max_diff # Scale same as original...

    w=5000
    h=5000
    fig=plt.figure(figsize=(32, 32))
    columns = 2
    rows = 5

    ax = fig.add_subplot(rows, columns, 1)
    show_sample(x[3])
    ax.set_title("Reference X")
    fig.add_subplot(rows, columns, 2)
    x_ref = x[3]
    img_hist(x_ref)

    ax = fig.add_subplot(rows, columns, 3)
    show_sample(y_hat)
    ax.set_title("Predicted Y")
    fig.add_subplot(rows, columns, 4)
    img_hist(y_hat)

    ax = fig.add_subplot(rows, columns, 5)
    show_sample(res)
    ax.set_title("Residual")
    ax = fig.add_subplot(rows, columns, 6)
    plt.hist(res_vals)
    ax.set_title(f"Mean {res_vals.mean()}, Std: {res_vals.std()}")

    ax = fig.add_subplot(rows, columns, 7)
    show_sample(y)
    ax.set_title("Ground Truth Y")
    fig.add_subplot(rows, columns, 8)
    img_hist(y)

    ax = fig.add_subplot(rows, columns, 9)
    show_sample(diff)
    ax.set_title("Ground Truth Difference")
    ax = fig.add_subplot(rows, columns, 10)
    plt.hist(diff_vals)
    ax.set_title(f"Mean {diff_vals.mean()}, Std: {diff_vals.std()}")

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(name)

mpl.use('Agg')
plt.ioff()

for i in range(11):
    visual_results(np.load(f'test{i}result.npz'), f'test{i}result.png')