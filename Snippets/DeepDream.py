import tensorflow as tf

class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def tracing(self, img, steps, stepSize):
        loss = tf.constant(0.0)

        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)

                activ = self.model(tf.expand_dims(img, axis=0))

                if len(activ) == 1:
                    activ = [activ]

                losses = []
                for act in activ:
                    loss = tf.math.reduce_mean(act)
                    losses.append(loss)

                loss = tf.reduce_sum(losses)

            gradients = tape.gradient(loss, img)

            gradients /= tf.math.reduce_std(gradients) + 1e-8

            img = tf.clip_by_value((img + gradients * stepSize), -1, 1)

        return loss, img
