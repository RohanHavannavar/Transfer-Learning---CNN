#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning

# Download all the data in this <a href='https://drive.google.com/open?id=1Z4TyI7FcFVEx8qdl4jO9qxvxaqLSqoEu'>rar_file</a> , it contains all the data required for the project.
#  When you unrar the file you'll get the files in the following format: <b>path/to/the/image.tif,category</b>
#             
#     where the categories are numbered 0 to 15, in the following order:
# <pre>
#     <b>0 letter
#     1 form
#     2 email
#     3 handwritten
#     4 advertisement
#     5 scientific report
#     6 scientific publication
#     7 specification
#     8 file folder
#     9 news article
#     10 budget
#     11 invoice
#     12 presentation
#     13 questionnaire
#     14 resume
#     15 memo</b>
#     
# </pre>

# There is a file named as 'labels_final.csv' , it consists of two columns. First column is path which is the required path to the images and second is the class label.

# In[ ]:


#!gdown --id 1Z4TyI7FcFVEx8qdl4jO9qxvxaqLSqoEu


# In[ ]:


# Method -2 you can also import the data using wget function
#https://www.youtube.com/watch?v=BPUfVq7RaY8
get_ipython().system('wget --header="Host: storage.googleapis.com" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,es;q=0.8" --header="Referer: https://www.kaggle.com/" "https://storage.googleapis.com/kaggle-data-sets/836734/1428684/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220309%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220309T112427Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=708c604ffc34e5957a02884a3894f5ffd58720ec81ab701353343b4d1a6ce20a0070613e95dc196928592e171b006f5cba83b700a09a3ab39265b44148a1a171a27a73262dae1e063afec7e2f4fec540ea769d5b9f44a4cc670ba4ac25cab93ee48915864bb1006656d5fae2947c16477a90eec3139aa1b3445aa3229eeddafc9179ac67617c73a05ede345fdaa60521618d29e6e41cf85fedcbebfbf7d54bdd55a5c8747f037f96285feb169f2934fde87ff1e924985b3bbdec501ae276802ebfef5615816c6b8e9bc2d815301fef29a77b1c70fbe487f8702d7c9d7a5969ac59cf5ee953c09e8bada1deb6cfd52ea67c1c706e773429404d1bfa98e744fc28" -c -O \'archive.zip\'')


# In[ ]:


#unrar the file
get_ipython().system("unzip '/content/archive.zip' -d '/content/'")


# ## 2. On this image data, you have to train 3 types of models as given below You have to split the data into Train and Validation data.

# In[ ]:


#import all the required libraries
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
df=pd.read_csv('labels_final.csv',dtype=str)


# In[ ]:


df.head(10)


# In[ ]:


import pathlib


# In[ ]:


dir_path = 'data_final'
data_root = pathlib.Path(dir_path)
print(data_root)


# In[ ]:


#converting tiff images to jpeg images
#https://stackoverflow.com/questions/28870504/converting-tiff-to-jpeg-in-python
import os, sys
from PIL import Image
for infile in df['path']:
    infile = os.path.join(data_root,infile)
    print("file : " + infile)
    if(infile[-3:] == "tif" or infile[-3:] == "bmp"):
       # print "is tif or bmp"
       outfile = infile[:-3] + "jpeg"
       im = Image.open(infile)
       print("new filename : " + outfile)
       out = im.convert("RGB")
       out.save(outfile, "JPEG", quality=90)


# In[ ]:


get_ipython().system('find . -type f -iname \\*.tif -delete')


# In[ ]:


#converting dataframe paths extension to jpeg
df['path'] = df['path'].apply(lambda x: x[:-3] + "jpeg")


# In[ ]:


df['path'] = df['path'].apply(lambda x : os.path.join(data_root,x))


# In[ ]:


df['path']


# In[ ]:


import random
all_image_paths = list(data_root.glob('*/*/*/*/*'))
all_image_paths = [str(path) for path in all_image_paths]

image_count = len(all_image_paths)
image_count


# In[ ]:


from PIL import Image
#using the tfdata_generator function given in reference notebook
def tfdata_generator(images, labels, is_training, batch_size=32):
    '''Construct a data generator using tf.Dataset'''
    
    def parse_function(filename, label):
        #reading path 
        image_string = tf.io.read_file(filename)
        #decoding image
        image = tf.image.decode_jpeg(image_string,channels = 3)
        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        #resize the image
        image = tf.image.resize(image, [224, 224])
        #one hot coding for label
        y = tf.one_hot(tf.cast(label, tf.uint8), 16)
        return image, y
    
    ##creating a dataset from tensorslices
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if is_training:
        dataset = dataset.shuffle(5000)  # depends on sample size

    # Transform and batch data at the same time
    dataset = dataset.apply(tf.data.experimental.map_and_batch( parse_function, batch_size,num_parallel_batches=4,  # cpu cores
        drop_remainder=True if is_training else False))
    

    dataset = dataset.cache('./tf-data')
    
    #repeat the dataset indefinitely
    dataset = dataset.repeat()

    
    #prefetch the data into CPU/GPU
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


# In[ ]:


#converting labels to int
df['label'] = pd.to_numeric(df['label'],downcast = 'integer')


# In[ ]:


#splitting data to train and test
train = df.sample(frac = 0.7,random_state = 200)
validation = df.drop(train.index)


# In[ ]:


#creating train and validation generator
train_generator = tfdata_generator(train['path'],train['label'],is_training=True, batch_size=32)
validation_generator = tfdata_generator(validation['path'],validation['label'],is_training=True, batch_size=32)


# In[ ]:


train_generator


# In[ ]:


validation_generator


# In[ ]:



train_steps_per_epoch = np.ceil(33600 / 32)
test_steps_per_epoch = np.ceil(14400/32)


# In[ ]:


print("training steps per epoch it means the batch number for train is",train_steps_per_epoch)
print("testing steps per epoch it means the batch number for test is",test_steps_per_epoch)


# In[ ]:


tf.keras.backend.clear_session()

## Set the random seed values to regenerate the model.
np.random.seed(0)


# In[ ]:



from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.callbacks import Callback
from keras.callbacks import TensorBoard


# ### Model-1

# <pre>
# 1. Use <a href='https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16'>VGG-16</a> pretrained network without Fully Connected layers and initilize all the weights with Imagenet trained weights. 
# 2. After VGG-16 network without FC layers, add a new Conv block ( 1 Conv layer and 1 Maxpooling ), 2 FC layers and an output layer to classify 16 classes. You are free to choose any hyperparameters/parameters of conv block, FC layers, output layer. 
# 3. Final architecture will be <b>INPUT --> VGG-16 without Top layers(FC) --> Conv Layer --> Maxpool Layer --> 2 FC layers --> Output Layer</b>
# 4.Print model.summary() and plot the architecture of the model. 
# <a href='https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model'>Reference for plotting model</a>
# 5. Train only new Conv block, FC layers, output layer. Don't train the VGG-16 network. 
# 
# </pre>

# In[ ]:


get_ipython().system('rm -rf ./logs/ ')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
import datetime
logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # tensorboard
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)


# In[ ]:


model = VGG16(input_shape =(224,224,3),weights = 'imagenet',include_top = False)


# In[ ]:


model.summary()


# In[ ]:


for layer in model.layers:
  layer.trainable = False

x = model.output
conv_1 = Conv2D(filters=512,kernel_size=(3,3),data_format='channels_last',padding="same", activation="relu")(x)
max_1 = MaxPool2D(2,2)(conv_1)
flat = Flatten()(max_1)
fc1 = Dense(256, activation="relu")(flat)
fc2 = Dense(128, activation="relu")(fc1)
output = Dense(16, activation="softmax")(fc2)

model_1 = Model(inputs = model.input, outputs = output)

model_1.compile(loss = "categorical_crossentropy", optimizer ='Adam', metrics=["accuracy"])


# In[ ]:


model_1.fit_generator(train_generator,steps_per_epoch = train_steps_per_epoch,epochs = 5,validation_data=validation_generator,validation_steps=test_steps_per_epoch,callbacks=[tensorboard_callback])


# In[ ]:


model_1.summary()


# In[ ]:


#plot the model
tf.keras.utils.plot_model(model_1, to_file='model_1.png', show_shapes=True, show_layer_names=True)


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs')


# ### Model-2

# <pre>
# 1. Use <a href='https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16'>VGG-16</a> pretrained network without Fully Connected layers and initilize all the weights with Imagenet trained weights.
# 2. After VGG-16 network without FC layers, don't use FC layers, use conv layers only as Fully connected layer.Any FC 
# layer can be converted to a CONV layer. This conversion will reduce the No of Trainable parameters in FC layers. 
# For example, an FC layer with K=4096 that is looking at some input volume of size 7×7×512 can be equivalently expressed as a CONV layer with F=7,P=0,S=1,K=4096. 
# In other words, we are setting the filter size to be exactly the size of the input volume, and hence the output will
# simply be 1×1×4096 since only a single depth column “fits” across the input volume, giving identical result as the 
# initial FC layer. You can refer <a href='http://cs231n.github.io/convolutional-networks/#convert'>this</a> link to better understanding of using Conv layer in place of fully connected layers.
# 3. Final architecture will be VGG-16 without FC layers(without top), 2 Conv layers identical to FC layers, 1 output layer for 16 class classification. <b>INPUT --> VGG-16 without Top layers(FC) --> 2 Conv Layers identical to FC -->Output Layer</b>
# 4. 4.Print model.summary() and plot the architecture of the model. 
# <a href='https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model'>Reference for plotting model</a>
# 5. Train only last 2 Conv layers identical to FC layers, 1 output layer. Don't train the VGG-16 network. 
# </pre>

# In[ ]:


get_ipython().system('rm -rf ./logs/ ')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
import datetime
logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # tensorboard
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)


# In[ ]:


#model_2
for layer in model.layers:
  layer.trainable = False
#converting FC layers to conv block so less weights are generated
x = model.output
in_1 = Conv2D(filters=256,kernel_size=7 ,data_format='channels_last',strides=1,activation="relu")(x)
in_2 = Conv2D(filters=256,kernel_size=1 ,data_format='channels_last',strides=1,activation="relu")(in_1)
in_3 = Flatten()(in_2)

output= Dense(16, activation="softmax")(in_3)
model_2 = Model(inputs = model.input, outputs = output)
# compile the model 
model_2.compile(loss="categorical_crossentropy",optimizer = 'Adam',metrics=['accuracy'])


# In[ ]:


model_2.fit_generator(train_generator,steps_per_epoch = train_steps_per_epoch,epochs = 3,validation_data=validation_generator,validation_steps=test_steps_per_epoch,callbacks=[tensorboard_callback])


# In[ ]:


model_2.summary()


# In[ ]:


tf.keras.utils.plot_model(model_2, to_file='model_2.png', show_shapes=True, show_layer_names=True)


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs')


# ### Model-3

# <pre>
# 1. Use same network as Model-2 '<b>INPUT --> VGG-16 without Top layers(FC) --> 2 Conv Layers identical to FC --> Output Layer</b>' and train only Last 6 Layers of VGG-16 network, 2 Conv layers identical to FC layers, 1 output layer.
# </pre>

# In[ ]:


get_ipython().system('rm -rf ./logs/ ')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
import datetime
logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # tensorboard
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)


# In[ ]:


for layer in model.layers[-6:]:
  layer.trainable = True

x = model.output
x = Conv2D(filters=256,kernel_size=7 ,data_format='channels_last',strides=1,activation="relu")(x)
x = Conv2D(filters=256,kernel_size=1 ,data_format='channels_last',strides=1,activation="relu")(x)
x = Flatten()(x)
 
output = Dense(16, activation="softmax")(x)
model_3 = Model(inputs = model.input, outputs = output)

model_3.compile(loss="categorical_crossentropy",optimizer = 'Adam',metrics=['accuracy'])


# In[ ]:


model_3.fit_generator(train_generator,steps_per_epoch = train_steps_per_epoch,epochs = 3,validation_data=validation_generator,validation_steps=test_steps_per_epoch,callbacks=[tensorboard_callback])


# In[ ]:


model_3.summary()


# In[ ]:


tf.keras.utils.plot_model(model_3, to_file='model_3.png', show_shapes=True, show_layer_names=True)


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs')


# ### Observations

# ##Model_1
# 1.Out of all parameters only some are trainable.
# 2.After 5 epochs we are getting almost 73% accuracy if we increase the epochs we can get more accuracy and less loss.
# 

# ##Model_2
# 1.Parameters or weights are less as compared to model_1 because last FC layers are converted to convolutional blocks.
# 
# 2.In one epoch model gave 68 percent accuracy. If we perrform more epochs the accuracy will be better

# #Model_3
# 
# 1.In model_1 and model_2 we are not training the layers of VGG 16 but in Model_3 we are training the last 6 layers of model.
# 
# 2.Accuracy is lower than model_1 and model_2
# 
# 3.Trainable parameters are more.

# In[ ]:


#summarising the model
from prettytable import PrettyTable
  

myTable = PrettyTable(["Models", "accuracy"])
  

myTable.add_row(["Model_1", "80%"])
myTable.add_row(["Model_2", "75%"])
myTable.add_row(["Model_3", "6%"])

  
  
print(myTable)

