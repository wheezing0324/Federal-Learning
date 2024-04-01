"""
    FedAvg args
"""

import argparse
import torch

#这段代码定义了一个函数args_parser，该函数使用argparse模块创建命令行参数解析器，并添加了多个参数选项。这些参数选项包括：
def args_parser():
    parser = argparse.ArgumentParser()
    #--E：本地模型训练次数，默认值为40。
    parser.add_argument('--E', type=int, default=20, help='本地模型训练次数')
    #--r：全局训练次数，默认值为12
    parser.add_argument('--r', type=int, default=10, help='全局训练次数')
    #--K：客户端总数，默认值为10。
    parser.add_argument('--K', type=int, default=5, help='客户端总数')
    #--input_dim：输入维度，默认值为8。
    parser.add_argument('--input_dim', type=int, default=8, help='输入维度')
    #--lr：学习率，默认值为0.01。
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    #--C：抽样率，默认值为0.5。
    parser.add_argument('--C', type=float, default=0.5, help='抽样率')
    #--B：本地批量大小，默认值为500。
    parser.add_argument('--B', type=int, default=200, help='本地批量大小')
    #--optimizer：优化器，默认为'adam'。
    parser.add_argument('--optimizer', type=str, default='adam', help='优化器')
    #--device：设备，默认为根据CUDA是否可用选择'cuda'或'cpu'。
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    #--weight_decay：权重衰减，默认值为1e-4。
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减，每轮全球学习率下降')
    #--clients：客户端列表，根据循环生成名为'diabetes1'到'diabetes10'的客户端名称。
    clients = ['diabetes' + str(i) for i in range(1, 6)]
    parser.add_argument('--clients', default=clients)

    args = parser.parse_args()

    return args
