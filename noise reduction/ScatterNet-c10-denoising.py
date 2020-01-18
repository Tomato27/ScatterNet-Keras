import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Concatenate, Conv2DTranspose, Add
from keras.layers.core import Dropout, Lambda
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.merge import concatenate
import os
import pickle
import numpy as np

#parameters
batch_size = 32
num_classes = 10
num_epochs = 50
use_resblock=True

model1_om="adadelta"
model1_lf="mean_squared_error"
model2_om="adam"
model2_lf="binary_crossentropy"

use_skip_connections=False

saveDir = "/opt/files/python/transfer/ae/uns"
if not os.path.isdir(saveDir):
      os.makedirs(saveDir)


#load cifar-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#devide val,test
x_val = x_test[:9000]
x_test = x_test[9000:]

def ConvertBlock(x): 
  x = Conv2D(8, (3, 3), padding='same')(x)
  x = PReLU()(x)
  #x = Dropout(0.1)(x)
  x = Conv2D(1, (3, 3),padding='same') (x)
  x = PReLU()(x)
  return x

def ResBlock(x,o_size):
  shortcut = x
  x = Conv2D(o_size, (3, 3), padding='same')(x)
  x = PReLU()(x)
  #x = Dropout(0.1)(x)
  x = Conv2D(o_size, (3, 3), padding='same') (x)

#####Residual#######
  x=keras.layers.Add()([x, shortcut])
# relu is performed right after each batch normalizatio
# expect for the output of the block where relu is performed after the adding to the shortcut
  x = PReLU()(x)
  return x

def DownBlock(x,o_size):  
  x = Conv2D(o_size, (2, 2),strides=2, padding='same')(x)
  x = PReLU()(x)
  x = ResBlock(x,o_size)
  return x

def UpBlock(x,o_size):
  x = UpSampling2D((2, 2))(x) #change it to bilinear upsampling
  x = Conv2D(o_size, (3, 3), padding='same')(x)
  x = PReLU()(x)
  x = ResBlock(x,o_size) 
  return x


#network architecture
input_img = Input(shape=(32, 32, 3))

# Build U-Net model
s = Lambda(lambda x: x / 255) (input_img)

cb = ConvertBlock(s)

rb = ResBlock(cb, 8)

db1 = DownBlock(rb, 16)

db2 = DownBlock(db1, 32)

db3 = DownBlock(db2, 64)

db4 = DownBlock(db3, 128)

db5 = DownBlock(db4, 256)

ub1 = UpBlock(db5, 128)

ub2 = concatenate([ub1, db4])
ub2 = UpBlock(ub2,64)

ub3 = concatenate([ub2, db3])
ub3 = UpBlock(ub3,32)

ub4 = concatenate([ub3, db2])
ub4 = UpBlock(ub4,16)

ub5 = concatenate([ub4, db1])
ub5 = UpBlock(ub5,8)

ob = concatenate([ub5, rb])
#outputs = Conv2D(3, (1, 1), activation='sigmoid')(ob)
outputs = Conv2D(3, (1, 1), activation=PReLU())(ob)

model = Model(input_img, outputs)

#binary crossentropy,MSEでの評価用モデル生成
model1 = Model(input_img,outputs)
model1.compile(optimizer=model1_om, loss=model1_lf)
model1.summary()

model2 = Model(input_img, outputs)
model2.compile(optimizer=model2_om, loss=model2_lf)

from keras.utils.vis_utils import plot_model
plot_model(model1,to_file="unet_s.png",show_shapes=1)

es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
chkpt = saveDir + 'UNET_Cifar10_s_Deep_weights2.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

#lerning
history = model1.fit(x_train, x_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    validation_data=(x_val, x_val),
                    #callbacks=[es_cb, cp_cb],
                    shuffle=True)

#evaluate
score = model1.evaluate(x_test, x_test, verbose=1)
print("score1:")
print(score)
score2 = model2.evaluate(x_test, x_test, verbose=1)
print("score2:")
print(score2)

#show_imgs
import matplotlib.pyplot as plt
#%matplotlib inline

decoded_imgs = model1.predict(x_test)

# utility function for showing images
fig1=plt.figure(figsize=(20, 4))
for i in range(10):
      ax = plt.subplot(2, 10, i+1)
      plt.imshow(x_test[i].reshape(32,32,3))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if decoded_imgs is not None:
        ax = plt.subplot(2, 10, i+ 1 +10)
        plt.imshow(decoded_imgs[i].reshape(32,32,3))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
#plt.show()


#plot transition of learn
print("Training history")
fig2 = plt.figure(figsize=(5,4))
plt.plot(model1.history.history['loss'],linestyle="solid",marker="o",label="training loss")
plt.plot(model1.history.history['val_loss'],linestyle="solid",marker="^",color="r",label="validation loss")
plt.title('training & validation loss')
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
