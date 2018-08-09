"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.batch = 100

        ##################
        # epsilon 的最小值，当 epsilon 小于该值时，将不在随机选择行为。
        self.FINAL_EPSILON = 0.0001

        # epsilon 的初始值，epsilon 逐渐减小。
        self.INITIAL_EPSILON = 0.25
        self.epsilon = self.INITIAL_EPSILON

        # epsilon 衰减的总步数。
        self.EXPLORE = 600000.
        #####################

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        ######
        norm_obs = self.tf_obs * 5.0
        ######
        layer = tf.layers.dense(
            inputs=norm_obs,
            units=10,
            activation=tf.nn.sigmoid,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),
            bias_initializer=tf.constant_initializer(0),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),
            bias_initializer=tf.constant_initializer(0),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability
        self.all_act = self.all_act_prob

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
            self.loss = loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def choose_action_test(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.argmax([prob_weights])#np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())  # select action w.r.t the actions prob

        return action

    def prob_result(self,feature):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs:feature})
        return prob_weights

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    # def learn(self):
    #     # discount and normalize episode reward
    #     discounted_ep_rs_norm = self._discount_and_norm_rewards()
    #
    #     # train on episode
    #     self.sess.run(self.train_op, feed_dict={
    #          self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
    #          self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
    #          self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
    #     })
    #     self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
    #     return discounted_ep_rs_norm

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # 一轮中包含的batch个数
        n_batch = int(len(discounted_ep_rs_norm) / self.batch)  # train on episode
        for batch_i in range(n_batch):
            batch_ep_obs = self.ep_obs[batch_i * self.batch:(batch_i + 1) * self.batch]
            batch_ep_as = np.array(self.ep_as)[batch_i * self.batch:(batch_i + 1) * self.batch]
            batch_ep_rs_norm = discounted_ep_rs_norm[batch_i * self.batch:(batch_i + 1) * self.batch]
            self.sess.run(self.train_op, feed_dict={
                 self.tf_obs: np.vstack(batch_ep_obs),  # shape=[None, n_obs]
                 self.tf_acts: np.array(batch_ep_as),  # shape=[None, ]
                 self.tf_vt: batch_ep_rs_norm,  # shape=[None, ]
            })
            if batch_i % 100 == 0:
                losss = self.sess.run([self.loss], feed_dict={#,self.all_act
                    self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
                    self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
                    self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
                })
                print(batch_i,losss)
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        # Return an array of zeros with the same shape and type as a given array
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def plot_cost(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(self.cost_his)), self.cost_his)
        ax.set(xlabel='training steps', ylabel='Cost')
        fig.savefig('Cost' + "_one.png")


