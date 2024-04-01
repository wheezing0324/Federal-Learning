"""
    FedAvg 客户端
"""

from data_process import dataSet
import numpy as np

def train(args, nn, file_name, num):
    #打印当前客户端的训练信息。
    print('Client', num + 1, 'training:')
    #从数据处理模块中获取训练数据集X_train、X_test、y_train和y_test
    X_train, X_test, y_train, y_test = dataSet(file_name, args.B)
    #设置模型的大小为训练数据集的长度。
    nn.len = len(X_train)  # 设置模型大小
    #打印模型大小信息。
    print("len=",nn.len)

    batch_size = args.B  # 本地批量大小-200
    epochs = args.E  # 本地模型训练次数-10
    print("batch_size=", batch_size)

    # 将整数类别y_train转换为二进制矩阵，其中num_classes=2表示有2个类别。
    #y_train_binary = to_categorical(y_train,num_classes=2)
    # 不指定压缩轴，让NumPy自动选择
    #y_train_binary = np.squeeze(y_train_binary)
    #使用训练数据集X_train和转换后的目标变量y_train_binary训练模型nn，并指定训练次数和批量大小。
    nn.fit(X_train, y_train ,epochs=epochs, batch_size=batch_size)
    #返回训练后的模型nn
    return nn

#这个函数接受参数args和nn，从数据处理模块中获取测试数据集
def test(args, nn):
    #从数据处理模块中获取测试数据集X_train、X_test、y_train和y_test
    X_train, X_test, y_train, y_test = dataSet(nn.file_name, args.B)
    #使用测试数据集X_test和y_test对模型nn进行评估，计算损失和准确率。
    loss, acc = nn.evaluate(
        X_test,
        y_test,
        batch_size=args.B,
        verbose=0
        #return_metrics=True
    )
    print("\nTest accuracy: %.3f%%" % (100.0 * acc))
    #print("Precision: %.3f" % precision)
    #print("Recall: %.3f" % recall)
    #print("F1 score: %.3f" % f1_score)


    return acc
