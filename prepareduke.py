import os

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 12:50:05 2018
@author: youxinlin
"""
import os


# # 返回原始图像路径名称
# def img_file_name(file_dir):
#     L = ''
#     for root, dirs, files in os.walk(file_dir):
#         for file in files:
#             if file == 'img.png':
#                 L = os.path.join(root, file)
#     #                print(L)
#     #                file_name = file[0:-4]  #去掉.png后缀
#     #                L.append(file_name)
#     #                L.append(' '+'this is anohter file\'s name')
#     return L


imgdir = r'E:\workspace_python\dataset\DukeMTMC-reID\bounding_box_train'
list_txt_file = r'E:\workspace_python\dataset\DukeMTMC-reID\bounding_box_train\train_list.txt'

docs = os.listdir(imgdir)  # 找出文件夹下所有的文件
print(docs)
for name in docs: # 找到每个_json结尾的文件夹
        print(name)
        ID = name.split('_')
        label = ID[0]
        print(label)
        with open(list_txt_file, 'a') as f:
            f.write(name + ' ' + label + '\n')
        f.close()
