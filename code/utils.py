import numpy as np
from gensim.models import Word2Vec


def get_w2v_features(data, word_index, id_word_index, vocab_size, id_vocab_size, args):
    w2v_model = Word2Vec(sentences=data['seq_list'].tolist(), vector_size=args.w2v_embed_size, window=5, min_count=1,
                         epochs=10)

    w2v_embedding_matrix = np.zeros((vocab_size, args.w2v_embed_size))
    for word, i in word_index.items():
        try:
            embedding_vector = w2v_model.wv.get_vector(word)
        except KeyError:
            continue
        if embedding_vector is not None:
            w2v_embedding_matrix[i] = embedding_vector

    id_model = Word2Vec(sentences=data['id_list'].tolist(), vector_size=args.id_embed_size, window=2, min_count=1,
                        epochs=10)

    id_embedding_matrix = np.zeros((id_vocab_size, args.id_embed_size))
    for word, i in id_word_index.items():
        try:
            embedding_vector = id_model.wv.get_vector(word)
        except KeyError:
            continue
        if embedding_vector is not None:
            id_embedding_matrix[i] = embedding_vector

    featureDict = {
        'g': [[0, 22, 30, 34], 57, 75, 156, 0.102, 0.085, 0.190, 0.152, 75.06714, 6.06, 2.34, 9.60, 48, 249.9, 0],
        # Glycine 甘氨酸
        'p': [[1, 26, 30, 34], 57, 55, 152, 0.102, 0.301, 0.034, 0.068, 115.13194, 6.30, 1.99, 10.96, 90, 1620.0,
              10.87],  # Proline 脯氨酸
        't': [[2, 27, 31, 34], 83, 119, 96, 0.086, 0.108, 0.065, 0.079, 119.12034, 5.60, 2.09, 9.10, 93, 13.2, 1.67],
        # Threonine 苏氨酸
        'e': [[3, 21, 29, 33], 151, 37, 74, 0.056, 0.060, 0.077, 0.064, 147.13074, 3.15, 2.10, 9.47, 109, 8.5, 2.09],
        # Glutamic Acid 谷氨酸
        's': [[4, 27, 31, 34], 77, 75, 143, 0.120, 0.139, 0.125, 0.106, 105.09344, 5.68, 2.21, 9.15, 73, 422.0, 1.25],
        # Serine 丝氨酸
        'k': [[5, 25, 32, 35], 114, 74, 101, 0.055, 0.115, 0.072, 0.095, 146.18934, 9.60, 2.16, 9.06, 135, 739.0,
              5.223888888888889],  # Lysine 赖氨酸
        'c': [[6, 28, 30, 34], 70, 119, 119, 0.149, 0.050, 0.117, 0.128, 121.15404, 5.05, 1.92, 10.70, 86, 280, 4.18],
        # Cysteine 半胱氨酸
        'l': [[7, 22, 30, 34], 121, 130, 59, 0.061, 0.025, 0.036, 0.070, 131.17464, 6.01, 2.32, 9.58, 124, 21.7, 9.61],
        # Leucine 亮氨酸
        'm': [[8, 28, 30, 34], 145, 105, 60, 0.068, 0.082, 0.014, 0.055, 149.20784, 5.74, 2.28, 9.21, 124, 56.2, 5.43],
        # Methionine 蛋氨酸
        'v': [[9, 22, 30, 34], 106, 170, 50, 0.062, 0.048, 0.028, 0.053, 117.14784, 6.00, 2.29, 9.74, 105, 58.1, 6.27],
        # Valine 缬氨酸
        'd': [[10, 21, 29, 33], 67, 89, 156, 0.161, 0.083, 0.191, 0.091, 133.10384, 2.85, 1.99, 9.90, 96, 5.0, 2.09],
        # Asparagine 天冬氨酸
        'a': [[11, 22, 30, 34], 142, 83, 66, 0.06, 0.076, 0.035, 0.058, 89.09404, 6.01, 2.35, 9.87, 67, 167.2, 2.09],
        # Alanine 丙氨酸
        'r': [[12, 25, 32, 35], 98, 93, 95, 0.070, 0.106, 0.099, 0.085, 174.20274, 10.76, 2.17, 9.04, 148, 855.6,
              5.223888888888889],  # Arginine 精氨酸
        'i': [[13, 22, 30, 34], 108, 160, 47, 0.043, 0.034, 0.013, 0.056, 131.17464, 6.05, 2.32, 9.76, 124, 34.5,
              12.54],  # Isoleucine 异亮氨酸
        'n': [[14, 23, 31, 34], 101, 54, 146, 0.147, 0.110, 0.179, 0.081, 132.11904, 5.41, 2.02, 8.80, 91, 28.5, 0],
        # Aspartic Acid 天冬酰胺
        'h': [[15, 24, 25, 32, 33], 100, 87, 95, 0.140, 0.047, 0.093, 0.054, 155.15634, 7.60, 1.80, 9.33, 118, 41.9,
              2.09],  # Histidine 组氨酸
        'f': [[16, 24, 30, 34], 113, 138, 60, 0.059, 0.041, 0.065, 0.065, 165.19184, 5.49, 2.20, 9.60, 135, 27.6,
              10.45],  # Phenylalanine 苯丙氨酸
        'w': [[17, 24, 30, 34], 108, 137, 96, 0.077, 0.013, 0.064, 0.167, 204.22844, 5.89, 2.46, 9.41, 163, 13.6,
              14.21],  # Tryptophan 色氨酸
        'y': [[18, 24, 31, 34], 69, 147, 114, 0.082, 0.065, 0.114, 0.125, 181.19124, 5.64, 2.20, 9.21, 141, 0.4, 9.61],
        # Tyrosine 酪氨酸
        'q': [[19, 23, 31, 34], 111, 110, 98, 0.074, 0.098, 0.037, 0.098, 146.14594, 5.65, 2.17, 9.13, 114, 4.7, -0.42],
        # Glutamine 谷氨酰胺
        'x': [[20], 99.9, 102.85, 99.15, 0.0887, 0.08429999999999999, 0.0824, 0.0875, 136.90127, 6.027,
              2.1690000000000005, 0.0875, 109.2, 232.37999999999997, 5.223888888888889],
        'u': [[], 99.9, 102.85, 99.15, 0.08870000000000001, 0.08430000000000001, 0.0824, 0.08750000000000001, 169.06,
              6.026999999999999, 2.1690000000000005, 9.081309523809526, 109.19999999999999, 232.37999999999997,
              5.223888888888889],
        'z': [[], 99.9, 102.85, 99.15, 0.08870000000000001, 0.08430000000000001, 0.0824, 0.08750000000000001,
              136.90126999999998, 6.026999999999999, 2.1690000000000005, 9.081309523809526, 109.19999999999999,
              232.37999999999997, 5.223888888888889],
    }

    fearure_embedding_matrix_one_hot = np.zeros((vocab_size, 36))
    fearure_embedding_matrix_numerical = np.zeros((vocab_size, 14))
    for word, i in word_index.items():
        feature_idx = featureDict.get(word)
        fearure_embedding_matrix_one_hot[i, feature_idx[0]], fearure_embedding_matrix_numerical[i] = 1, feature_idx[1:]
    mean, std = fearure_embedding_matrix_numerical.mean(axis=0), fearure_embedding_matrix_numerical.std(axis=0)
    fearure_embedding_matrix_numerical = (fearure_embedding_matrix_numerical - mean) / (std + 1e-10)
    feature_embedding_matrix = np.concatenate([fearure_embedding_matrix_one_hot, fearure_embedding_matrix_numerical],
                                              axis=1)

    return w2v_embedding_matrix, id_embedding_matrix, feature_embedding_matrix