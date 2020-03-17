import random;
import numpy as np
import pandas as pd
import tensorflow as tf;
import tensorflow.keras.backend as K
#K.set_learning_phase(1)
n=np.array([[[[1],[3],[3]],[[1],[3],[3]]]],dtype=float)
n=np.append(n,[[[[3],[40],[3]],[[2],[2],[2]]]],axis=3)
b=np.array([[[[3],[2],[2]],[[2],[2],[2]]]],dtype=float)
b=np.append(b,[[[[9],[90],[9]],[[0],[0],[0]]]],axis=3)
c=np.append(n,b,axis=0)
a=np.array([[1,3,3],[2,2,2],[4,4,4]],dtype=float)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.BatchNormalization(axis=-1,input_shape=(2,3,2),trainable=True,scale=False))
model.compile(loss='mean_absolute_error',optimizer='adam')
print(np.shape(n))
print(model(c,training=True))