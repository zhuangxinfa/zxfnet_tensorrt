# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:44:11 2020

@author: zhuan
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np


x = tf.placeholder(tf.float32,shape = [10,10,1],name = "x_placeholder")
two = tf.constant(2.0,dtype = tf.float32,shape=[10,10,1],name = "two_op")
y=tf.add(x,two,name = "add_op")
sum = tf.mat5
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(y,feed_dict={x:np.ones(shape=[10,10,1], dtype = float)}))
# print(sess.run(two))
graph_def = tf.get_default_graph().as_graph_def()

# convert_variables_to_constants()函数表示用相同值的常量替换计算图中所有变量，
# 原型convert_variables_to_constants(sess,input_graph_def,output_node_names,
#                          variable_names_whitelist, variable_names_blacklist)
# 其中sess是会话，input_graph_def是具有节点的GraphDef对象，output_node_names
# 是要保存的计算图中的计算节点的名称，通常为字符串列表的形式，variable_names_whitelist
# 是要转换为常量的变量名称集合(默认情况下，所有变量都将被转换)，
# variable_names_blacklist是要省略转换为常量的变量名的集合。
output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def,output_node_names=["add_op"])
with tf.gfile.GFile("./test2x.pb", "wb") as f:
            # SerializeToString()函数用于将获取到的数据取出存到一个string对象中，
            # 然后再以二进制流的方式将其写入到磁盘文件中
                f.write(output_graph_def.SerializeToString())