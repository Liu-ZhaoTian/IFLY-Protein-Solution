from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()

    parser.add_argument('--w2v_embed_size', type=int, default=78)
    parser.add_argument('--feature_embed_size', type=int, default=50)
    parser.add_argument('--id_embed_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=300)

    args = parser.parse_args()
    return args
