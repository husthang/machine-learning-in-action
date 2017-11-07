# coding=utf-8
"""
Decision Tree source code for machine learning in action Ch. 03

@file: trees.py
@time: 2017/11/6 下午2:42
@author: liuhang
@email: liuhang93@foxmail.com
"""

from math import log
import operator


def create_data_set():
    """
    创建简单的数据集    
    :return:返回数据集 
    """
    data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def cal_shannon_entropy(data_set):
    """
    计算给定数据集的香农熵，function to calculate the Shannon entropy of a data set
    熵的理解：熵用来衡量一个随机变量的不确定性，熵越大，不确定性越高，越无序；熵越小，不确定性越小，越有序，携带的信息越容易被确定。熵可用比特做单位，可理解为平均编码长度
    :return: 香农熵 shannon_entropy
    """
    num_entries = len(data_set)
    label_counts = {}
    for entry in data_set:
        current_label = entry[-1]
        label_counts[current_label] = label_counts.get(current_label, 0) + 1
    shannon_entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy


def split_data_set(data_set, axis, value):
    """
    按照给定特征值划分数据集，datas et splitting on a given feature,
    :param data_set:待划分的数据集 
    :param axis: 划分数据集的特征
    :param value: 特征的返回值
    :return: 得到第axis个特征值为value时的子数据集
    """
    ret_data_set = []
    for entry in data_set:
        if entry[axis] == value:
            entry_temp = entry[:axis]
            entry_temp.extend(entry[axis + 1:])
            ret_data_set.append(entry_temp)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    Choosing the best feature to split on，选择出数据集中，最能区分类别的那个特征值
    :param data_set: 
    :return: 信息增益最大的特征的编号
    """
    features_num = len(data_set[0]) - 1
    base_entropy = cal_shannon_entropy(data_set)
    best_info_gain, best_feature = 0.0, -1
    for i in range(features_num):
        val_list = [entry[i] for entry in data_set]
        unique_vals = set(val_list)  # 得到第i个特征的所有可能取值
        new_entropy = 0.0
        # 下面for循环计算第i个特征值的条件熵
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)  # 第i个特征值，取值为value的子特征集
            prob = len(sub_data_set) / float(len(data_set))  # 第i个特征值，取值为value的概率
            new_entropy += prob * cal_shannon_entropy(sub_data_set)
        # 计算得到第i个特征值的信息增益，信息增益越大，说明这个特征值确定时，系统的不确定性下降的多，即这个特征值用来区分系统的能力越强
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def major_count(class_list):
    """
    计算类别中，出现次数最多的那个类别
    :param class_list: 
    :return: 出现次数最多的类别
    """
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    递归创建决策树，得到一个嵌套的字典结构
    :param data_set: 数据集
    :param labels: 特征的名称
    :return: 返回决策树
    """
    class_list = [entry[-1] for entry in data_set]
    if class_list.count(class_list[0]) == len(class_list):  # 类别完全相同，停止划分，返回这个类别
        return class_list[0]
    if len(data_set[0]) == 1:  # 已经遍历完所有特征，此时类别还没有完全相同，返回集合中次数出现最多的类别
        # pass
        return major_count(class_list)
    best_feat_index = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat_index]
    labels.pop(best_feat_index)
    my_tree = {best_feat_label: {}}
    unique_feat_values = set([entry[best_feat_index] for entry in data_set])
    for value in unique_feat_values:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat_index, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vector):
    """
    使用决策树的分类函数
    :param input_tree: 输入的决策树 
    :param feat_labels: 特征标签名称
    :param test_vector: 待分类的数据
    :return: 得到的分类类别
    """
    current_label = input_tree.keys()[0]
    feat_index = feat_labels.index(current_label)
    feat_value = test_vector[feat_index]
    sub_nodes = input_tree[current_label]
    key = sub_nodes[feat_value]
    if isinstance(key, dict):
        class_label = classify(key, feat_labels, test_vector)
    else:
        class_label = key
    return class_label


if __name__ == '__main__':
    pass