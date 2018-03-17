
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import requests
import urllib
from PIL import Image 
import os
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# In[ ]:

#Get image from url

#a = 1
#with open('Cat_image.txt','r') as f:
#  urls = []
#  for url in f:
#      urls.append(url.strip())
#      try:
#          with urllib.request.urlopen(url) as url_:
#            try:
#              with open('temp.jpg', 'wb') as f:
#                f.write(url_.read())
#              img = Image.open('temp.jpg')
#              name = "test{}.jpg".format(a)
#              img.save(name)
#              a += 1
#            except:
#                pass
#      except:
#        pass
#print("done")
#print(a)


# In[ ]:

## resize image to 28x28

#count = range(0,1033)
#for i in count:
#    cat1 = Image.open('cat ({}).jpg'.format(i))
#    new_image = cat1.resize((28,28))
#    new_image.save('cat{}.jpg'.format(i))
#
#print('done')


# In[2]:

train = []
validation = []
test = []

##Get cat image##
os.chdir("C:\\Users\\USER\\python studyspace\\Deep learning\\Project\\cat_32")
print(os.getcwd())

#add cat image to train_set --> size 1200 
for i in range(1,1201):  
    pic = Image.open('cat{}.jpg'.format(i))
    pix = np.array(pic)
    train.append(pix)
#train_set = np.array(train)

#add cat image to validation_set --> size 200
for i in range(1201,1401):
    pic = Image.open('cat{}.jpg'.format(i))
    pix = np.array(pic)
    validation.append(pix)
#validation_set = np.array(validation)    

#add cat image to test_set --> size 200
for i in range(1401,1601):
    pic = Image.open('cat{}.jpg'.format(i))
    pix = np.array(pic)
    test.append(pix)
#test_set = np.array(test)

### Get horse image
os.chdir("C:\\Users\\USER\\python studyspace\\Deep learning\\Project\\monkey_32")
print(os.getcwd())

#add monkey image to train_set --> size 900 
for j in range(1,901):  
    pic = Image.open('monkey{}.jpg'.format(j))
    pix = np.array(pic)
    train.append(pix)
    #print(train)
train_set = np.array(train)

#add monkey image to validation_set --> size 200
for j in range(901,1101):
    pic = Image.open('monkey{}.jpg'.format(j))
    pix = np.array(pic)
    validation.append(pix)
validation_set = np.array(validation)    

#add monkey image to test_set --> size 200
for j in range(1101,1301):
    pic = Image.open('monkey{}.jpg'.format(j))
    pix = np.array(pic)
    test.append(pix)
test_set = np.array(test)

os.chdir("C:\\Users\\USER\\python studyspace\\Deep learning\\Project")


# In[3]:

print(train_set.shape)
print(validation_set.shape)
print(test_set.shape)


# In[4]:

plt.imshow(train_set[0]) # cat image example


# In[5]:

plt.imshow(train_set[1600]) # monkey image example


# In[ ]:

#change into gray image
#train_set[[0],:,:,[2]] =train_set[[0],:,:,[0]]
#train_set[[0],:,:,[1]] = train_set[[0],:,:,[0]]
#plt.imshow(train_set[0])


# In[4]:


# Set train_labels
train_labels = np.zeros((2100))
train_labels[0:1200] = 0   ## 0 == cat
train_labels[1200:2100] = 1 ## 1 == monkey

# Set validation labels
validation_labels = np.zeros((400))
validation_labels[0:200] = 0  ## 0 == cat
validation_labels[200:600] = 1  ## 1 == monkey

#Set Test labels
test_labels = np.zeros((400))
test_labels[0:200] = 0  ## 0 == cat
test_labels[200:400] =1  ## 1 == monkey


# In[5]:

#Shuffle dataset & labels

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_set, train_labels = randomize(train_set, train_labels)
validation_set, validation_labels = randomize(validation_set, validation_labels)
test_set, test_labels = randomize(test_set, test_labels)


# In[6]:

num_labels =2
image_size = 32
num_channels = 3  
## cause RGB image

## reformat all data set & labels

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size,image_size,num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_set, train_labels = reformat(train_set, train_labels)
validation_set, validation_labels = reformat(validation_set, validation_labels)
test_set, test_labels = reformat(test_set, test_labels)
print('train_set : ',train_set.shape, train_labels.shape)
print('validation_set : ',validation_set.shape, validation_labels.shape)
print('test_set : ',test_set.shape, test_labels.shape)


# In[11]:

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels,1))
          / predictions.shape[0])


# In[9]:

batch_size = 128
learning_rate = 0.001
patch_size = 7
depth = 64
num_hidden = 128


graph = tf.Graph()
with graph.as_default():
    
  tf_train_dataset = tf.placeholder(tf.float32,
                                     shape=[None,image_size , image_size,3],name = 'train_dataset')
  tf_train_labels = tf.placeholder(tf.float32, shape=[None, num_labels], name = 'train_label')
  tf_valid_dataset = tf.constant(validation_set)
  tf_test_dataset = tf.constant(test_set)
    
  ## Setting First Layer
  ## so w_conv1 has 64 filter which is 7x7x3 shape
  W_conv1 = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    # depth means number of filters
  b_conv1 = tf.Variable(tf.zeros([depth]))

  ##Setting Second Layer
  W_conv2 = tf.Variable(tf.truncated_normal(
       [patch_size, patch_size, depth, depth], stddev = 0.1))
  b_conv2 = tf.Variable(tf.zeros([depth]))
    
  ## Setting First FC Layer
  W_fc1 = tf.Variable(tf.truncated_normal(
        [image_size//4 * image_size // 4 * depth, num_hidden],stddev=0.1))
  b_fc1 = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

  ## Setting Second FC Layer
  W_fc2 = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
  b_fc2 = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  def set_model(data):
    L_conv1 = tf.nn.conv2d(data, W_conv1, [1,1,1,1], padding='SAME')
    L_conv1 = tf.nn.relu(L_conv1+b_conv1)
    
    #pooling
    #pooling has no parameters to learn --> fixed function
    L_conv1 = tf.nn.max_pool(L_conv1, ksize=[1,3,3,1],
                       strides=[1,2,2,1], padding='SAME')
    #Normalization
    L_conv1 = tf.nn.lrn(L_conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
 
    
    #L1 = tf.nn.dropout(L1, keep_prob = 0.7)
    L_conv2 = tf.nn.conv2d(L_conv1,W_conv2, [1,1,1,1], padding='SAME')
    L_conv2 = tf.nn.relu(L_conv2+b_conv2)
    

    #pooling
    L_conv2 = tf.nn.max_pool(L_conv2, ksize=[1,3,3,1],
                       strides=[1,2,2,1], padding='SAME')
    #Normalization
    L_conv2 = tf.nn.lrn(L_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
   
    #L2 = tf.nn.dropout(L2, keep_prob = 0.7)

    shape = L_conv2.get_shape().as_list()
    reshape = tf.reshape(L_conv2, [-1, shape[1] * shape[2] * shape[3]])
    
    L_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1)+b_fc1)
    #L3 = tf.nn.dropout(L3, keep_prob = 0.7)
    return tf.matmul(L_fc1, W_fc2) + b_fc2
  

  logits = set_model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits= logits))
  
  optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)

  # y_pred = tf.nn.softmax(logits, name='y_pred')
  train_prediction = tf.nn.softmax(logits, name='train_pred')
  valid_prediction = tf.nn.softmax(set_model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(set_model(tf_test_dataset))
                   

        


# In[12]:

num_steps = 1001

with tf.Session(graph=graph) as session:
  saver = tf.train.Saver(tf.global_variables())
  '''  ckpt = tf.train.get_checkpoint_state('./model')
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(session, ckpt.model_checkpoint_path)
  else:
    session.run(tf.global_variables_initializer())'''
  session.run(tf.global_variables_initializer())
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_set[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), validation_labels))
  saver.save(session, "./save2.ckpt")
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# In[ ]:



