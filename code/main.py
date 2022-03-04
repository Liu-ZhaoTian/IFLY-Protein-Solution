import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score

from model import my_model
from args import make_args
from utils import get_w2v_features
from data import get_data, read_fasta_file


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return


def main():
    args = make_args()
    train = read_fasta_file('../data/astral_train.fa', 9472)
    test = read_fasta_file('../data/astral_test.fa', 2371)
    data, number_2_number, X_train, X_test, X_train_id, X_test_id, word_index, vocab_size, id_word_index, \
    id_vocab_size, y_categorical = get_data(train, test, args.max_len)

    w2v_embedding_matrix, id_embedding_matrix, feature_embedding_matrix = get_w2v_features(data, word_index,
                                                                                           id_word_index, vocab_size,
                                                                                           id_vocab_size, args)

    # 五折交叉验证
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    oof = np.zeros([X_train.shape[0], 245])
    predictions = np.zeros([X_test.shape[0], 245])

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, train['label'])):
        model = my_model(vocab_size, id_vocab_size, w2v_embedding_matrix, id_embedding_matrix, feature_embedding_matrix,
                         args)
        if fold_ == 0:
            print(model.summary())

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1', patience=5, mode='max')
        bst_model_path = f"../model/{fold_}.h5"
        ck_callback = tf.keras.callbacks.ModelCheckpoint(bst_model_path,
                                                         monitor='val_f1',
                                                         mode='max', verbose=2,
                                                         save_best_only=True,
                                                         save_weights_only=True)

        X_tra, X_val = X_train[trn_idx], X_train[val_idx]
        X_tra_id, X_val_id = X_train_id[trn_idx], X_train_id[val_idx]
        y_tra, y_val = y_categorical[trn_idx], y_categorical[val_idx]

        print("fold n{}".format(fold_ + 1))
        model.fit([X_tra, X_tra_id], y_tra,
                  validation_data=([X_val, X_val_id], y_val),
                  epochs=args.epoch, batch_size=args.batch_size, shuffle=True,
                  callbacks=[Metrics(valid_data=([X_val, X_val_id], y_val)), early_stopping, ck_callback])

        model.load_weights(bst_model_path)

        oof[val_idx] = model.predict([X_val, X_val_id])

        predictions += model.predict([X_test, X_test_id]) / folds.n_splits
        del model

    xx_cv = f1_score(train['label'].values, np.argmax(oof, axis=1), average='macro')
    print("F1", xx_cv)

    result = pd.DataFrame()
    result['sample_id'] = test['id'].copy()
    result['category_id'] = np.argmax(predictions, axis=1)
    result['category_id'] = result['category_id'].map(number_2_number)
    result['category_id'] = result['category_id'].apply(lambda x: '.'.join(str(x).split('.')[:2]))
    print(result)


if __name__ == "__main__":
    main()
