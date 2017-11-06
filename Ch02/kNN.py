# coding=utf-8
"""
kNN: k Nearest Neighbors, K邻近算法

@file: kNN.py
@time: 2017/11/6 上午10:31
@author: liuhang
@email: liuhang93@foxmail.com
"""
import numpy as np
import operator
import matplotlib.pyplot as plt


def create_data_set():
    """
    创建数据集的测试
    :return: 数据矩阵和标签向量
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(input_vector, data_set, labels, k):
    """
    分类器
    :param input_vector: 用于分类的输入向量
    :param data_set: 输入的训练样本集
    :param labels: 类别标签向量
    :param k:  k邻近算法，最邻近的数目
    :return: 发生频率最高的标签
    """
    data_set_size = data_set.shape[0]  #
    diff_mat = np.tile(input_vector, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indices = distances.argsort()  # 将数组排序后，返回相应位置的元素在原数组中的索引
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indices[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    """
    从文本中解析数据
    :param filename: 文件名
    :return:数据矩阵，分类标签向量 
    """
    with open(filename) as f:
        lines_array = f.readlines()
    lines_num = len(lines_array)
    return_mat = np.zeros((lines_num, 3))
    index = 0
    class_label_vector = []
    for line in lines_array:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    """
    归一化数值
    :param data_set: 输入数据集
    :return: 输出数据集
    """
    min_vals = data_set.min(0)  # 求第0轴的最小值
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    line_num = data_set.shape[0]  # 第0轴数量，行数
    norm_data_set = data_set - np.tile(min_vals, (line_num, 1))  # 贴
    norm_data_set = norm_data_set / np.tile(ranges, (line_num, 1))
    return norm_data_set


def dating_class_test():
    """
    分类器针对约会数据的测试
    :return: 输出测试结果及错误率
    """
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    rows_num = dating_data_mat.shape[0]
    norm_mat = auto_norm(dating_data_mat)
    num_test_vecs = int(rows_num * 0.5)
    error_count = 0
    for i in range(num_test_vecs):
        classifier_result = classify(norm_mat[i, :], norm_mat[num_test_vecs:rows_num, :],
                                     dating_labels[num_test_vecs:rows_num], 3)
        print 'the classifier came back with: %d, the real answer is %d ' % (classifier_result, dating_labels[i])
        if classifier_result != dating_labels[i]:
            error_count += 1
    print 'the total number of errors is %d ' % error_count
    print 'the total error rate is: %f' % (error_count / float(num_test_vecs))


def mat_plot_test():
    """
    使用matplotlib分析数据
    :return: 画出散点图
    """
    dating_data_mat, dating_labels = file2matrix('datingTestSet1.txt')
    plt.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2], 15.0 * np.array(dating_labels),
                15.0 * np.array(dating_labels))
    plt.show()


if __name__ == '__main__':
    dating_class_test()
