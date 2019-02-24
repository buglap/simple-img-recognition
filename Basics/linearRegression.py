# Supervised learning
# simplest machine learning models (y =mx +b), we the objective
# o fit some data points to help with predictions

# Program optimize line by adjusting m and b until minimize loss

import tensorflow as tf

#we need to provide some expected data "y"
#y= mx +b
#x= [1,2,3,4]
#y =[0,-1,-2,-3]

m = tf.Variable([-.5],dtype = tf.float32)
b = tf.Variable([.5],dtype = tf.float32)

x = tf.placeholder(dtype = tf.float32)
y = tf.placeholder(dtype = tf.float32)

linear_model = m*x + b

#for know how bad is our model
loss= tf.reduce_sum(tf.square(linear_model - y))

x_train = [1,2,3,4]
#[0,-0.5,-1,-1.5] first obtained

y_train = [0,-1,-2,-3]
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
# print(session.run(loss, {x: x_train, y: y_train}))

##training our model, for minimize the loss (close to zero)

#learning rate as a parameter, it has to be a tiny value for garantice a convergention
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
    session.run(train,{x: x_train, y: y_train})

new_m, new_b, new_loss = session.run([m,b,loss],{x: x_train, y: y_train})
# print("New m: %s"%new_m)
# print("New b: %s"%new_b)
# print("New loss: %s"%new_loss)

print(session.run(linear_model, {x: [10, 20, 30, 40]}))