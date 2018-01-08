import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

np.random.seed(0)

data1 = np.random.rand(50, 2) + np.array([0.7, 0.0])
data2 = np.random.rand(50, 2)

def calc_w(data1, data2):
    """2クラスの線型判別分析における、写像ベクトルwを計算する関数。

    wは、はじパタP.81 (6.26)式における最大固有値に対応する固有ベクトルとして計算する。

    Args:
        data1 (numpy.ndarray): 1クラス目に属するデータのデータ行列。shape=(N1, d)
        data2 (numpy.ndarray): 2クラス目に属するデータのデータ行列。shape=(N2, d)

    Return:
        d次元空間から1次元空間への写像を行う写像ベクトルw
    """
    ### TASK ###
    pass

def draw_figure(data_list, filename, flag=False):
    color = ["blue", "red", "blue", "red"]
    shape = ["*", "+", "*", "+"]

    for i, data in enumerate(data_list):
        x = data[:, 0]
        if len(data[0]) == 1:
            y = np.zeros(len(data))
        else:
            y = data[:, 1]
        plt.scatter(x, y, c=color[i], marker=shape[i])

    plt.savefig(filename)
    plt.show()

def main():
    w = calc_w(data1, data2)
    data1_1d = data1 @ w[np.newaxis].T
    data2_1d = data2 @ w[np.newaxis].T
    draw_figure([data1, data2, data1_1d, data2_1d], "lda.png", flag=True)

if __name__ == "__main__":
    main()
