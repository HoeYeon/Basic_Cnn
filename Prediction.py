
# coding: utf-8

# In[2]:

import numpy as np
import tensorflow as tf
import requests
import urllib
from PIL import Image 
import os
import matplotlib.pyplot as plt
import cv2 as cv2

get_ipython().magic('matplotlib inline')


# In[3]:

os.chdir("C:\\Users\\USER\\python studyspace\\Deep learning\\Project")
pic = Image.open("cat_test.jpg")
new_image = pic.resize((32,32))
test1 = np.array(new_image)
test1 = test1.reshape(1,32,32,3)
print(test1.shape)


# In[5]:

plt.imshow(pic)


# In[6]:

sess = tf.Session()

saver = tf.train.import_meta_graph('save2.ckpt.meta')

saver.restore(sess, tf.train.latest_checkpoint('./'))
    
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("train_pred:0")

x = graph.get_tensor_by_name("train_dataset:0")
y_true = graph.get_tensor_by_name("train_label:0")

y_test_images = np.zeros((1,2))

feed_dict_testing = {x: test1, y_true: y_test_images}

result=sess.run(y_pred, feed_dict=feed_dict_testing)


# In[7]:

print(result)


# In[ ]:



