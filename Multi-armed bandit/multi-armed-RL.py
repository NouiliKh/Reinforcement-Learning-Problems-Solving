import tensorflow as tf
import numpy as np

bandits = [0.1,0,-0.5,-8]
numb_bandits = len(bandits)

def pull_bandit(bandit):
    # A random number
    random_number = np.random.randn(1)
    if random_number > bandit:
        return 1
    else:
        return -1

tf.reset_default_graph()

#initialize bandits weights to 1
weights = tf.Variable(tf.ones([numb_bandits]))
#position of the max bandit values
#axis = 0 ==> rows
chosen_bandit = tf.argmax(weights,axis=0)

reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)

responsible_weight = tf.slice(weights,action_holder,[1])

#caluculating loss value with log loss fucntion
loss =-(tf.log(responsible_weight)*reward_holder)

#optimize the value
Optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
update=Optimizer.minimize(loss)

total_eppch =1000
total_reward =np.zeros(numb_bandits)
e=0.1

init=tf.initialize_all_variables()

#lauch tf graph

with tf.Session() as sess :
    sess.run(init)
    i=0
    while i< total_eppch :
        if np.random.rand(1)< e:
            action = np .random.randint(numb_bandits)
        else :
            action = sess.run(chosen_bandit)

        reward = pull_bandit(bandits[action])

        _, resp, ww = sess.run([update, responsible_weight, weights],
                               feed_dict={reward_holder: [reward], action_holder: [action]})

        # Update our running tally of scores.
        total_reward[action] += reward
        if i % 50 == 0:
            print
            "Running reward for the " + str(numb_bandits) + " bandits: " + str(total_reward)
        i += 1
    print
    "The agent thinks bandit " + str(np.argmax(ww) + 1) + " is the most promising...."
    if np.argmax(ww) == np.argmax(-np.array(bandits)):
        print
        "...and it was right!"
    else:
        print
        "...and it was wrong!"