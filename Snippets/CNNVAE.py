import numpy as np
import tensorflow as tf


class CNNVAE(object):
    def __init__(
        self,
        zSize=32,
        batchSize=1,
        lr=0.0001,
        tolerence=0.5,
        training=False,
        reuse=False,
    ):
        self.zSize = zSize
        self.batchSize = batchSize
        self.lr = lr
        self.tolerence = tolerence
        self.training = training
        self.reuse = reuse

        with tf.variable_scope("CNNVAE", reuse=self.reuse):
            self.build()

        self._init_session()

    def build(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

            encoder = tf.reshape(
                tf.layers.conv2d(
                    tf.layers.conv2d(
                        tf.layers.conv2d(
                            tf.layers.conv2d(
                                self.x,
                                32,
                                4,
                                strides=2,
                                activation=tf.nn.relu,
                                name="conv1",
                            ),
                            64,
                            4,
                            strides=2,
                            activation=tf.nn.relu,
                            name="conv2",
                        ),
                        128,
                        4,
                        strides=2,
                        activation=tf.nn.relu,
                        name="conv3",
                    ),
                    256,
                    4,
                    strides=2,
                    activation=tf.nn.relu,
                    name="conv4",
                ),
                [-1, 2 * 2 * 256],
            )

            self.mu = tf.layers.dense(encoder, self.zSize, name="mu")
            self.sigma = tf.exp(
                tf.layers.dense(encoder, self.zSize, name="logVar") / 2.0
            )
            self.epsilon = tf.random_normal([self.batchSize, self.zSize])

            self.z = self.mu + self.sigma * self.epsilon

            decoder = tf.layers.conv2d_transpose(
                tf.layers.conv2d_transpose(
                    tf.layers.conv2d_transpose(
                        tf.reshape(
                            tf.layers.dense(self.z, 1024, name="fc"), [-1, 1, 1, 1024],
                        ),
                        128,
                        5,
                        strides=2,
                        activation=tf.nn.relu,
                        name="deconv1",
                    ),
                    64,
                    5,
                    strides=2,
                    activation=tf.nn.relu,
                    name="deconv2",
                ),
                32,
                6,
                strides=2,
                activation=tf.nn.relu,
                name="deconv3",
            )

            self.y = tf.layers.conv2d_transpose(
                decoder, 3, 6, strides=2, activation=tf.nn.sigmoid, name="deconv4",
            )

            if self.training:
                self.step = tf.Variable(0, name="step", trainable=False)

                self.rLoss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.square(self.x - self.y), reduction_indices=[1, 2, 3]
                    )
                )

                self.klLoss = tf.reduce_mean(
                    tf.maximum(
                        (
                            -0.5
                            * tf.reduce_sum(
                                (
                                    1
                                    + self.logvar
                                    - tf.square(self.mu)
                                    - tf.exp(self.logvar)
                                ),
                                reduction_indices=1,
                            )
                        ),
                        self.tolerence * self.zSize,
                    )
                )

                self.loss = self.rLoss + self.klLoss
                self.lr = tf.Variable(self.lr, trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.lr)

                grads = self.optimizer.compute_gradients(self.loss)

                self.trainOP = self.optimizer.apply_gradients(
                    grads, global_step=self.step, name="trainStep"
                )

            self.init = tf.global_variables_initializer()
