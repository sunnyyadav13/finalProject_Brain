# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:56:14 2022

@author: lenovo
"""

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from PIL import Image
from tensorflow.keras.models import load_model

from flask import Flask, request, render_template

from werkzeug.utils import secure_filename

app = Flask(__name__)

#model =load_model('vgg16_final.h5')

#model=tf.keras.models.load_model('E:/Codes/Project/finalProject_Brain/vgg16_final.h5')

def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor"
	elif classNo==1:
		return "Yes Brain Tumor"


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((224, 224))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    result=np.argmax(result,axis=1)
    return result