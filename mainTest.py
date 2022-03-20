import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('braintumor60epoch.h5')

image=cv2.imread('C:\\Users\\lenovo\\Downloads\\Major Project\\BrainTumor Classification DL\\pred\\pred0.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=model.predict_classes(input_img)
print(result)




