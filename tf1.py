import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
state = tf.Variable(0, name = 'counter')
# 定义变量state，赋值为0，并定义了一个名字叫：counter
print(state.name)
one = tf.constant(1)
# 定义一个常量1
new_value = tf.add(state, one)
update = tf.compat.v1.assign(state, new_value)
# 将new_value的值重新赋值给state
init = tf.compat.v1.global_variables_initializer()
# 初始化所有变量
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))