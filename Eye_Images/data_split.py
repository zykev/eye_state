# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 11:46:03 2021

@author: admin
"""

# train test set split for eye images

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

open_eye = os.listdir('E:/Dataset/Eye_Images/openLeftEyes')
close_eye = os.listdir('E:/Dataset/Eye_Images/closedLeftEyes')

data_eye = np.array(open_eye + close_eye)
label_eye = np.repeat([0,1],[len(open_eye),len(close_eye)])

data_train, data_test, labels_train, labels_test = train_test_split(data_eye, label_eye, stratify = label_eye, test_size=0.1, random_state = 0)

train_info = pd.DataFrame(data={'name': data_train, 'label': labels_train})
test_info = pd.DataFrame(data={'name': data_test, 'label': labels_test})

train_info.to_csv('E:/Dataset/Eye_Images/train_info.csv', sep=' ', index=False)
test_info.to_csv('E:/Dataset/Eye_Images/test_info.csv', sep=' ', index=False)