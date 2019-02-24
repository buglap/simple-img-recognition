#Explore constand and operator nodes
#How to set up and use nodes
#how to run nodes
import tensorflow as tf

const_node_1 = tf.constant(1.0, dtype =tf.float32)
const_node_2 = tf.constant(2.0, dtype= tf.float32)
const_node_3 = tf.constant([3.0,4.0,5.0], dtype =tf.float32)

adder_node_1 =tf.add(const_node_1, const_node_2)
adder_node_2 = const_node_1 + const_node_2

mult_node_1 = adder_node_2 * const_node_3

sess = tf.Session()
# print(sess.run(adder_node_1))
# print(sess.run(mult_node_1))

#Placeholder node, Nodes with no current value
#Pass in value when running session

placeholder_1 =tf.placeholder(dtype =tf.float32)
placeholder_2 = tf.placeholder(dtype =tf.float32)

#needs to proved a value, when use a placeholder
print(sess.run(placeholder_1, {placeholder_1:[3.0,8.0,2.0]}))

#example
multiply_node = placeholder_1 * placeholder_2

print(sess.run(multiply_node,{placeholder_1:4.0, placeholder_2:[3.0,5.4]}))

# #Variable node
# Store an initial value but can change
# Must call an initializer to assign the value

var_node_1 = tf.Variable([5.0], dtype = tf.float32)

#need for use variables
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(var_node_1))

#Have to run in session for re-assing a value
sess.run(var_node_1.assign([47]))

