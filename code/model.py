import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


class SelfAttentionDot(Layer):

    def __init__(self, attention_dim, **kwargs):
        self.attention_dim = attention_dim
        super(SelfAttentionDot, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.map_2_u = Dense(self.attention_dim)
        self.context_u = self.add_weight(shape=(1, self.attention_dim, 1),
                                         initializer='uniform',
                                         trainable=True)

        super(SelfAttentionDot, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        u_mat = self.map_2_u(x)
        batched_context_u = K.tile(self.context_u, (K.shape(x)[0], 1, 1))
        alignment = tf.einsum('ijk,ikl->ijl', u_mat, batched_context_u)
        alignment = K.softmax(alignment)
        result = tf.einsum('ijk,ikl->ijl', K.permute_dimensions(alignment, [0, 2, 1]), x)
        return tf.squeeze(result, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.attention_dim


def my_model(vocab_size, id_vocab_size, w2v_embedding_matrix, id_embedding_matrix, feature_embedding_matrix, args):
    seq_input = Input(shape=(args.max_len,), dtype='int32')
    id_input = Input(shape=(7,), dtype='int32')

    w2v_embedder = Embedding(vocab_size, args.w2v_embed_size, input_length=args.max_len,
                             weights=[w2v_embedding_matrix], trainable=True)
    w2v_embedding = w2v_embedder(seq_input)

    feature_embedder = Embedding(vocab_size, args.feature_embed_size, input_length=args.max_len,
                                 weights=[feature_embedding_matrix], trainable=False)
    feature_embedding = feature_embedder(seq_input)

    id_embedder = Embedding(id_vocab_size, args.id_embed_size, input_length=7, weights=[id_embedding_matrix],
                            trainable=True)

    id_embedding = id_embedder(id_input)
    id_embedding = Flatten()(id_embedding)
    embedding = concatenate([w2v_embedding, feature_embedding], axis=2)

    cnn_list = []
    for window in [1, 7, 9, 12, 24, 36]:
        cnn = Conv1D(window, window, padding='same', strides=1, activation='relu')(embedding)
        cnn_max = MaxPooling1D(pool_size=36)(cnn)
        cnn_mean = AveragePooling1D(pool_size=36)(cnn)
        cnn_list.append(cnn_max)
        cnn_list.append(cnn_mean)

    for windows in [1, 7, 9, 12, 24, 36]:
        cnn = Conv1D(36, windows, padding='same', strides=1, activation='relu')(embedding)
        cnn_max = MaxPooling1D(pool_size=36)(cnn)
        cnn_mean = AveragePooling1D(pool_size=36)(cnn)
        cnn_list.append(cnn_max)
        cnn_list.append(cnn_mean)

    reversed_cnn_list = []
    for window in [1, 7, 9, 12, 24, 36]:
        cnn = Conv1D(window, window, padding='same', strides=1, activation='relu')(tf.reverse(embedding, axis=[1]))
        cnn_max = MaxPooling1D(pool_size=36)(cnn)
        cnn_mean = AveragePooling1D(pool_size=36)(cnn)
        reversed_cnn_list.append(cnn_max)
        reversed_cnn_list.append(cnn_mean)

    for windows in [1, 7, 9, 12, 24, 36]:
        cnn = Conv1D(36, windows, padding='same', strides=1, activation='relu')(tf.reverse(embedding, axis=[1]))
        cnn_max = MaxPooling1D(pool_size=36)(cnn)
        cnn_mean = AveragePooling1D(pool_size=36)(cnn)
        reversed_cnn_list.append(cnn_max)
        reversed_cnn_list.append(cnn_mean)

    gru_embedding = Bidirectional(GRU(64, return_sequences=True))(embedding)
    att_embedding = SelfAttentionDot(128)(gru_embedding)

    all_embedding = concatenate(cnn_list + reversed_cnn_list, axis=2)

    flat_embedding = Flatten()(all_embedding)
    flat_embedding = concatenate([flat_embedding, id_embedding, att_embedding], axis=1)
    flat_embedding = BatchNormalization()(flat_embedding)
    flat_embedding = Dropout(0.2)(flat_embedding)
    output = Dense(245, activation='softmax')(flat_embedding)

    model = Model(inputs=[seq_input, id_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
