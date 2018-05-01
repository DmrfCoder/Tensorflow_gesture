# -*-coding:utf-8-*-


gamma=0.01
power=0.8


def adjust_learning_rate_inv(iter,base_lr):
    return base_lr * (1 + gamma * iter)**(- power)
