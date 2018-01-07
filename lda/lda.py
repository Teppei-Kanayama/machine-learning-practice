import numpy as np
import numpy.linalg as LA
import pdb

sample_data = np.array([[1.1, 2.2], [3.4, 4.1], [5.1, 6.1], [2.5, 3.1], [1.1, 4.1]])
sample_labels = np.array([1, 1, 0, 1, 0])

def split_data(data, labels, target_label):
    return data[np.where(labels == target_label)]

def calc_mean(data, labels, target_label):
    return np.mean(split_data(data, labels, target_label), axis=0)

def calc_sb(data, labels):
    mu = calc_mean(data, labels, 0) - calc_mean(data, labels, 1)
    return mu[np.newaxis].T @ mu[np.newaxis]

def calc_sk(data, labels, k):
    splited_data = split_data(data, labels, k)
    return np.cov(splited_data.T) * len(splited_data)

def calc_sw(data, labels):
    return calc_sk(data, labels, 0) + calc_sk(data, labels, 1)

def calc_w(data, labels):
    sb = calc_sb(data, labels)
    sw = calc_sw(data, labels)
    s = LA.inv(sw) @ sb
    rambda, v = LA.eig(s)
    return v[np.argmax(rambda)]

def main():
    print(calc_sb(sample_data, sample_labels))
    print(calc_sw(sample_data, sample_labels))
    print(lda(sample_data, sample_labels))

if __name__ == "__main__":
    main()
