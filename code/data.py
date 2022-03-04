import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import text, sequence


def read_fasta_file(x, number_sample=9472):
    seq = [0] * number_sample
    id = []
    with open(x, encoding='utf8') as f:
        for line in f:
            if line.startswith('>'):
                tmp = ''
                id.append(line)
            else:
                tmp = tmp + line.replace('\n', '')
            seq[len(id) - 1] = tmp
    data = pd.DataFrame()
    data['id'] = id
    data['seq'] = seq
    return data


def get_data(train, test, max_len):
    # 训练数据提取标签
    train['label'] = train['id'].apply(lambda x: str(x).split(' ')[1])
    train['id'] = train['id'].apply(lambda x: str(x).split(' ')[0].replace('>', ''))
    train['label'] = train['label'].apply(lambda x: '.'.join(str(x).split('.')[:2]))
    # 245 分类，多分类任务
    test['id'] = test['id'].apply(lambda x: str(x).replace('>', '').strip())

    train_label = list(train['label'].unique())
    label_2_number = dict(zip(train_label, list(range(0, len(train_label)))))
    number_2_number = dict(zip(list(range(0, len(train_label))), train_label))

    train['label'] = train['label'].map(label_2_number)
    data = pd.concat([train, test])

    data['seq_list'] = data['seq'].apply(lambda x: [i.lower() if i.lower() != 'z' else 'x' for i in x])
    data['id_list'] = data['id'].apply(lambda x: [i.lower() for i in x])
    X_train = data[~data['label'].isna()]['seq_list']
    X_test = data[data['label'].isna()]['seq_list']
    X_train_id = data[~data['label'].isna()]['id']
    X_test_id = data[data['label'].isna()]['id']
    y_categorical = to_categorical(train['label'].values, 245)

    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    print(f'vocab size: {vocab_size}')

    id_tokenizer = text.Tokenizer(char_level=True)
    id_tokenizer.fit_on_texts(list(X_train_id) + list(X_test_id))
    X_train_id = id_tokenizer.texts_to_sequences(X_train_id)
    X_test_id = id_tokenizer.texts_to_sequences(X_test_id)
    X_train_id = sequence.pad_sequences(X_train_id)
    X_test_id = sequence.pad_sequences(X_test_id)

    id_word_index = id_tokenizer.word_index
    id_vocab_size = len(id_word_index) + 1

    return data, number_2_number, X_train, X_test, X_train_id, X_test_id, word_index, vocab_size, id_word_index, id_vocab_size, y_categorical
