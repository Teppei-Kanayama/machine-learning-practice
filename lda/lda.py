import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import pdb

np.random.seed(0)
data1 = np.random.rand(50, 2) + np.array([1.2, 0.0])
data2 = np.random.rand(50, 2)

def calc_sb(data1, data2):
    mu = np.mean(data1, axis=0) - np.mean(data2, axis=0)
    return mu[np.newaxis].T @ mu[np.newaxis]

def calc_sk(data):
    return np.cov(data.T) * len(data)

def calc_sw(data1, data2):
    return calc_sk(data1) + calc_sk(data2)

def calc_w(data1, data2):
    sb = calc_sb(data1, data2)
    sw = calc_sw(data1, data2)
    s = LA.inv(sw) @ sb
    rambda, v = LA.eig(s)
    return v[np.argmax(rambda)]

def draw_figure(data_list, filename, w=None, flag=False):
    color = ["blue", "red", "blue", "red"]
    shape = ["*", "+", "*", "+"]

    for i, data in enumerate(data_list):
        x = data[:, 0]
        if len(data[0]) == 1:
            y = np.zeros(len(data))
        else:
            y = data[:, 1]
        plt.scatter(x, y, c=color[i], marker=shape[i])

    if w is not None:
        x = np.linspace(0,2,4)
        y = - (w[0] / w[1]) * x
        plt.plot(x,y,"r-")

    plt.savefig(filename)
    plt.show()

def main():
    w = calc_w(data1, data2)
    data1_1d = data1 @ w[np.newaxis].T
    data2_1d = data2 @ w[np.newaxis].T
    draw_figure([data1, data2, data1_1d, data2_1d], "after.png", w=w, flag=True)

if __name__ == "__main__":
    main()
