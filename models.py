import tensorflow as tf
import numpy as np


class Attention(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

        self.fe_layer = tf.keras.layers.Dense(units)
        self.hi_layer = tf.keras.layers.Dense(units)
        self.att_layer = tf.keras.layers.Dense(1)

    def call(self, features, hidden):

        attention_hidden_layer = tf.nn.tanh(
            self.fe_layer(features) + self.hi_layer(tf.expand_dims(hidden, 1))
        )

        score = self.att_layer(attention_hidden_layer)

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Encoder(tf.keras.Model):
    def __init__(self, encoder_dim):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.fc = tf.keras.layers.Dense(encoder_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, vocab_size, units):
        super().__init__()
        self.units = units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # self.embedding = tf.keras.layers.Embedding(
        #     vocab_size,
        #     embedding_dim,
        #     embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        #     trainable=False,
        # )
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc1 = tf.keras.layers.Dense(units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = Attention(self.units)

    def call(self, x, features, hidden):

        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        x = self.fc1(output)

        x = tf.reshape(x, (-1, x.shape[2]))

        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units), dtype=tf.float32)
