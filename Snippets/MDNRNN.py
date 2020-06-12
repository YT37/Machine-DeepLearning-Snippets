import numpy as np
import tensorflow as tf


class MDNRNN(object):
    def __init__(self, hps, reuse=False):
        self.hps = hps
        with tf.variable_scope("MDNRNN", reuse=reuse):
            self.graph = tf.Graph()

            with self.graph.as_default():
                self.build(hps)

        self._init_session()

    def build(self):
        KMIX = self.hps.numMixture
        INWIDTH = self.hps.inputWidth
        OUTWIDTH = self.hps.outputWidth
        LENGTH = self.hps.maxLen

        if self.hps.training:
            self.step = tf.Variable(0, name="step", trainable=False)

        cellfn = tf.contrib.rnn.LayerNormBasicLSTMCell

        if self.hps.recurrentDropout:
            cell = cellfn(
                self.hps.rnnSize,
                layer_norm=self.hps.layerNorm,
                dropout_keep_prob=self.hps.recurrentDropoutProb,
            )
        else:
            cell = cellfn(self.hps.rnnSize, layer_norm=self.hps.layerNorm)

        if self.hps.inputDropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, input_keep_prob=self.hps.inputDropoutProb
            )

        if self.hps.outputDropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=self.hps.outputDropoutProb
            )

        self.inputX = tf.placeholder(
            dtype=tf.float32, shape=[self.hps.batchSize, self.hps.maxLen, INWIDTH]
        )

        self.outputX = tf.placeholder(
            dtype=tf.float32, shape=[self.hps.batchSize, self.hps.maxLen, OUTWIDTH],
        )
        actualX = self.inputX

        self.initState = cell.zero_state(
            batch_size=self.hps.batchSize, dtype=tf.float32
        )

        NOUT = OUTWIDTH * KMIX * 3

        with tf.variable_scope("RNN"):
            outputW = tf.get_variable("outputW", [self.hps.rnnSize, NOUT])
            outputB = tf.get_variable("outputB", [NOUT])

        output, lastState = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=actualX,
            initial_state=self.initState,
            dtype=tf.float32,
            swap_memory=True,
            scope="RNN",
        )

        output = tf.reshape(
            tf.nn.xw_plus_b(
                tf.reshape(output, [-1, self.hps.rnnSize]), outputW, outputB
            ),
            [-1, KMIX * 3],
        )

        self.finalState = lastState

        outLogmix, outMean, outLogstd = tf.split(output, 3, 1)
        outLogmix = outLogmix - tf.reduce_logsumexp(outLogmix, 1, keepdims=True)
        lossfunc = -tf.reduce_mean(
            tf.reduce_logsumexp(
                (
                    outLogmix
                    + -0.5
                    * (
                        (tf.reshape(self.outputX, [-1, 1]) - outMean)
                        / tf.exp(outLogstd)
                    )
                    ** 2
                    - outLogstd
                    - np.log(np.sqrt(2.0 * np.pi))
                ),
                1,
                keepdims=True,
            )
        )

        if self.hps.training == 1:
            self.optimizer = tf.train.AdamOptimizer(
                tf.Variable(self.hps.learningRate, trainable=False)
            )

            cappedGVS = [
                (tf.clip_by_value(grad, -self.hps.gradClip, self.hps.gradClip), var)
                for grad, var in self.optimizer.compute_gradients(
                    tf.reduce_mean(lossfunc)
                )
            ]

            self.train_op = self.optimizer.apply_gradients(
                cappedGVS, global_step=self.step, name="trainStep"
            )

        self.init = tf.global_variables_initializer()
