import random;
import numpy as np
import pandas as pd
import tensorflow as tf;
d=pd.DataFrame([[1,2,3,4]])
c=d.values
a=pd.DataFrame([[2,3,4,5]])
mse=tf.keras.losses.MeanSquaredError()
print(mse(d.values,a.values).numpy())
