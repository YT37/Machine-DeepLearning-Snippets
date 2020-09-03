import bert
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp import optimization
from official.nlp.bert.tokenization import FullTokenizer

tokenizer = bert.bert_tokenization.FullTokenizer
bertLayer = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False
)
tokenizer = tokenizer(
    bertLayer.resolved_object.vocab_file.asset_path.numpy(),
    bertLayer.resolved_object.do_lower_case.numpy(),
)


print(tokenizer.tokenize("Roses are red, violets are blue."))
print(
    tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize("Roses are red, violets are blue.")
    )
)


def getIDs(tokens):
    return tokenizer.convert_tokens_to_ids(tokens)


def getMask(tokens):
    return np.char.not_equal(tokens, "[PAD]").astype(int)


def getSegments(tokens):
    ids = []
    currentId = 0
    for tok in tokens:
        ids.append(currentId)
        if tok == "[SEP]":
            currentId = 1 - currentId
    return ids


sent = tokenizer.tokenize("Roses are red, violets are blue.")

inputs = tf.expand_dims([getIDs(sent), getMask(sent), getSegments(sent)], axis=0)

inputs[:, 0, :]


class Model(tf.keras.Model):
    def __init__(self, units, dropout):
        super(Model, self).__init__()

        self.bertLayer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
            trainable=True,
        )

        self.dense = tf.keras.layers.Dense(
            units, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )(inputs)

    def call(self, inputs):
        sentence, token = self.embedder(
            [inputs[:, 0, :], inputs[:, 1, :], inputs[:, 2, :]]
        )
        token = tf.nn.dropout(token, self.dropout)
        output = self.dense(token)

        return output


units = 2
dropout = 0.1
batchSize = 32
epochs = 5
lr = 5e-5
warmupsteps = int(batchSize * 0.1)

model = Model(NB_UNITS, DROPOUT_RATE)

optimizer = optimization.create_optimizer(
    init_lr=lr, num_train_steps=batchSize, num_warmup_steps=warmupSteps
)


"""classifier""".compile(optimizer, """lossfn""", ["""metric"""])

