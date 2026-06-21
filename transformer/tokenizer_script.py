from transformers import FSMTTokenizer
from utils import WMTDataset
from tqdm import tqdm
import argparse
import transformers
import numpy as np
import pickle


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", default="allenai/wmt16-en-de-dist-6-1", type=str)
    parser.add_argument("--csv", default="./archive/wmt14_translate_de-en_train.csv", type=str)
    parser.add_argument("--data_out", default="tokenized_data.pkl", type=str)
    parser.add_argument("--meta_out", default="tokenized_meta.pkl", type=str)
    return parser


def main():
    args = get_parser().parse_args()

    transformers.utils.logging.set_verbosity_error()
    tokenizer = FSMTTokenizer.from_pretrained(args.tokenizer_name)
    dataset = WMTDataset(args.csv)

    tokenized_dataset = []
    lengths = []

    for data in tqdm(dataset):
        de_ids, en_ids = tokenizer(data).input_ids
        tokenized_dataset.append((de_ids, en_ids))
        lengths.append((len(de_ids), len(en_ids)))

    with open(args.data_out, 'wb') as f:
        pickle.dump(tokenized_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.meta_out, 'wb') as f:
        pickle.dump(np.array(lengths, dtype=np.int32), f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"wrote {len(tokenized_dataset)} pairs -> {args.data_out}, {args.meta_out}")


if __name__ == "__main__":
    main()
