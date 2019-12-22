import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import csv, ast
import time, os, sys
import pprint;
dir = os.getcwd()
data_dir = dir + "/Data"
league=['Bundesliga 1','Ligue 1','Premier League','Primera Division','Serie A']

def splitdata():
    global league
    for i in range(league.__len__()):
        with open('Data/FinalData/'+league[i]+' '+'2019'+'Gu clear NG.csv')as file:
            temp2019=pd.read_csv(file,header=None)
            if(i==0):
                data2019=temp2019.values
            else:
                data2019=np.concatenate((data2019,temp2019.values),axis=0)
            file.close()
    for j in range(league.__len__()):
        with open('Data/FinalData/'+league[j]+' '+'2018'+'Gu clear NG.csv')as file:
            temp2018=pd.read_csv(file,header=None)
            if (j == 0):
                traindata = temp2018.values
            else:
                traindata = np.concatenate((traindata, temp2018.values), axis=0)
            file.close()
        with open('Data/FinalData/'+league[j]+' '+'2017'+'Gu clear NG.csv')as file:
            temp2017=pd.read_csv(file,header=None)
            traindata=np.concatenate((traindata,temp2017.values),axis=0)
            file.close()
        with open('Data/FinalData/'+league[j]+' '+'2016'+'Gu clear NG.csv')as file:
            temp2016=pd.read_csv(file,header=None)
            traindata=np.concatenate((traindata,temp2016.values),axis=0)
            file.close()
    train_sample=traindata.copy()
    a=VarianceThreshold(threshold=0.07).fit_transform(train_sample)
    pprint.pprint(a)
    pprint.pprint(a.shape)
    np.random.shuffle(train_sample)
    test_sample,cv_sample=train_test_split(data2019,train_size=0.5,shuffle=True,random_state=0)

    train_sample_y_corner=train_sample[:,[68,69]].copy()
    cv_sample_y_corner=cv_sample[:,[68,69]].copy()
    test_sample_y_corner=test_sample[:,[68,69]].copy()
    train_sample_y_goal=train_sample[:,[70,71]].copy()
    cv_sample_y_goal=cv_sample[:,[70,71]].copy()
    test_sample_y_goal=test_sample[:,[70,71]].copy()
    train_sample_x=train_sample[:,0:68].copy()
    cv_sample_x=cv_sample[:,0:68].copy()
    test_sample_x=test_sample[:,0:68].copy()

    return train_sample_x,cv_sample_x,test_sample_x,train_sample_y_corner,cv_sample_y_corner,test_sample_y_corner,train_sample_y_goal,cv_sample_y_goal,test_sample_y_goal
train_x,cv_x,test_x,train_y_corner,cv_y_corner,test_y_corner,train_y_goal,cv_y_goal,test_y_goal=splitdata()
np.where(train_y_corner>10,10,train_y_corner)
np.where(cv_y_corner>10,10,cv_y_corner)
np.where(test_y_corner>10,10,test_y_corner)
train_y_corner_home=train_y_corner[:,0:1]
train_y_corner_away=train_y_corner[:,1:2]
train_y_goal_home=train_y_goal[:,0:1]
train_y_goal_away=train_y_goal[:,1:2]
cv_y_corner_home=cv_y_corner[:,0:1]
cv_y_corner_away=cv_y_corner[:,1:2]
cv_y_goal_home=cv_y_goal[:,0:1]
cv_y_goal_away=cv_y_goal[:,1:2]
test_y_corner_home=test_y_corner[:,0:1]
test_y_corner_away=test_y_corner[:,1:2]
test_y_goal_home=test_y_goal[:,0:1]
test_y_goal_away=test_y_goal[:,1:2]
def transferlabel(list):
    for i in range(len(list)):
        if(list[i]>0):
            list[i]=0
        elif(list[i]==0):
            list[i]=1
        elif(list[i]<0):
            list[i]=2
    return list
train_cornerwinner=train_y_corner_home-train_y_corner_away
print(type(train_cornerwinner))
cv_cornerwinner=cv_y_corner_home-cv_y_corner_away
test_cornerwinner=test_y_corner_home-test_y_corner_away
train_cornerwinner=transferlabel(train_cornerwinner)
cv_cornerwinner=transferlabel(cv_cornerwinner)
test_cornerwinner=transferlabel(test_cornerwinner)


initial_learning_rate=0.01
l1=5
l2=5
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=500,
    decay_rate=0.95,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
train_loss=0;
cv_loss=0;
test_loss=0;
train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
cv_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()

def create_sequential_model():
    model = tf.keras.models.Sequential()
    # Add layers
    model.add(tf.keras.layers.BatchNormalization(input_shape=(68,)))
    model.add(tf.keras.layers.Dense(16, activation="relu",kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1,l2=l2),use_bias=True))
    model.add(tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),use_bias=True))
    model.add(tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),use_bias=True))
    model.summary()
    return model
cost=tf.keras.losses.MeanAbsoluteError()
mean=tf.keras.metrics.Mean()
homemodel=create_sequential_model()
homemodel.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy() ,
    optimizer=optimizer,
    metrics=["accuracy"]
)
awaymodel=create_sequential_model()
awaymodel.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy() ,
    optimizer=optimizer,
    metrics=["accuracy"]
)
totalmodel=create_sequential_model()
totalmodel.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy() ,
    optimizer=optimizer,
    metrics=["accuracy"]
)
totalgoalmodel=create_sequential_model()
totalgoalmodel.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy() ,
    optimizer=optimizer,
    metrics=["accuracy"]
)
cornermodel=create_sequential_model()
cornermodel.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy() ,
    optimizer=optimizer,
    metrics=["accuracy"]
)
def train_step(inputs, targets,model):
    global train_loss
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = cost(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss=loss
    del tape
def validate_step(inputs_valid, targets_valid,model):
    global cv_loss
    predictions = model(inputs_valid)
    loss = cost(targets_valid, predictions)
    cv_loss=loss



def train_model(inputs, targets, inputs_valid, targets_valid,model,epochs):
    global train_loss,cv_loss
    graph_loss = []
    graph_validate = []
    for epoch in range(epochs):
        train_step(inputs, targets,model)
        validate_step(inputs_valid, targets_valid,model)
        print("epoch " + str(epoch))
        print("Loss:"+str(train_loss))
        print("Validate Loss:"+str(cv_loss))
        graph_loss.append(train_loss)
        graph_validate.append(cv_loss)

    plt.plot(graph_loss[10:], label="Train")
    plt.plot(graph_validate[10:], label="Valid")
    plt.title("Loss")
    plt.legend()
    plt.show()
def model_eval(inputs_test, targets_test,model):
    global test_loss
    predictions = model(inputs_test)
    loss = cost(targets_test, predictions)
    test_loss=loss
    print("Test Loss:"+str(test_loss))
def train_model_fit(x,y,model,epoch,cvx,cvy):
    h=model.fit(x,y,epochs=epoch,steps_per_epoch=48,validation_data=(cvx,cvy),shuffle=True)
    plt.plot(h.history['accuracy'],label="Train")
    plt.plot(h.history['accuracy'], label="Val")
    plt.legend()
    plt.show()


#train_model_fit(train_x,train_cornerwinner,cornermodel,500,cv_x,cv_cornerwinner)
#print(cornermodel.evaluate(test_x,test_cornerwinner))
train_model(train_x,train_y_corner_home+train_y_corner_away,cv_x,cv_y_corner_home+cv_y_corner_away,totalmodel,5000)
model_eval(test_x,test_y_corner_away+test_y_corner_home,totalmodel)
totalmodel.save("AItotal Zscore.h5")
'''


# The loss method
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# The optimize
optimizer = tf.keras.optimizers.Adam()
# This metrics is used to track the progress of the training loss during the training
train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='val_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='val_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


def create_sequential_model(number_of_outputs):
    # Flatten
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Flatten(input_shape=[784]))
    # Add layers
    model.add(tf.keras.layers.Dense(128, activation="sigmoid", input_shape=[290]))
    # model.add(tf.keras.layers.Dense(32, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(64, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(128, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(number_of_outputs, activation="softmax"))
    # model.summary()
    return model


model = create_sequential_model(6)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        print(inputs)
        print(predictions)
        loss = loss_object(targets, predictions)
        print(targets)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(targets, predictions)
    print(type(targets_test))


@tf.function
def validate_step(inputs_valid, targets_valid):
    predictions = model(inputs_valid)
    loss = loss_object(targets_valid, predictions)
    val_loss(loss)
    val_accuracy(targets_valid, predictions)


def reset_states():
    train_loss.reset_states()
    val_loss.reset_states()
    train_accuracy.reset_states()
    val_accuracy.reset_states()


def train_model(epochs):
    global inputs, targets, inputs_valid, targets_valid
    graph_loss = []
    graph_validate = []
    for epoch in range(epochs):
        train_step(inputs, targets)
        validate_step(inputs_valid, targets_valid)
        print("epoch " + str(epoch))
        print("Loss: {}, accuracy {}".format(train_loss.result(), train_accuracy.result() * 100))
        print("Validate Loss: {}, validation accuracy: {}".format(val_loss.result(), val_accuracy.result() * 100))
        graph_loss.append(train_loss.result().numpy())
        graph_validate.append(val_loss.result().numpy())
        reset_states()

    plt.plot(graph_loss, label="Train")
    plt.plot(graph_validate, label="Valid")
    plt.title("Loss")
    plt.legend()
    plt.show()


def model_eval(inputs_test, targets_test):
    predictions = model(inputs_test)
    loss = loss_object(targets_test, predictions)
    test_loss(loss)
    test_accuracy(targets_test, predictions)
    print(type(targets_test))
    print(targets_test.shape)
    print(type(predictions))
    print(predictions.shape)
    p = []
    pre = predictions.numpy()
    print(pre.shape[0])
    print(pre.shape[1])
    for i in range(pre.shape[0]):
        max = 0
        temp = 0
        for j in range(pre.shape[1]):
            if (pre[i][j] > max):
                max = pre[i][j]
                temp = j
        p.append(temp)

    numdefaut = 0;

    for k in range(len(targets_test)):
        if (targets_test[k] != p[k]):
            numdefaut = numdefaut + 1
            print(str(targets_test[k]) + '/' + str(pre[k]))
    print(numdefaut)
    print(numdefaut / len(targets_test))
    print("Test Loss: {}, acc: {}".format(test_loss.result(), test_accuracy.result()))
    print(len(pre))
    test_loss.reset_states()
    test_accuracy.reset_states()


train_model(800);
model_eval(inputs_test, targets_test)


# model.save(AI_dir+dataColor+'ALL_VECTORS_1_'+"AIhome.h5")
# model.load_weights(dir+'/AI/AIs/'+dataColor+'0_'+"AIhome.h5")
# model = tf.keras.models.load_model(AI_dir+dataColor+'ALL_VECTORS_1_'+"AIhome.h5")
'''