"""One-time migration: split the old combined pickle into clean files.

Old format (single file, two stacked dumps):
    pickle.dump(tokenized_dataset)   # list of [de_ids, en_ids]
    pickle.dump(metadata)            # list of (len_de, len_en)

New format (two files):
    tokenized_data.pkl  -> list of (de_ids, en_ids)
    tokenized_meta.pkl  -> np.int32 array of shape (N, 2)

Re-tokenization is NOT needed; this only re-packs the existing tokens.
"""
import argparse
import numpy as np
import os
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="tokenized_data.pkl", type=str,
                        help="existing combined pickle (data + metadata)")
    parser.add_argument("--data_out", default="tokenized_data.pkl", type=str)
    parser.add_argument("--meta_out", default="tokenized_meta.pkl", type=str)
    args = parser.parse_args()

    with open(args.src, 'rb') as f:
        tokenized_dataset = pickle.load(f)
        metadata = pickle.load(f)

    # Write the small metadata file first (cheap, lets us fail early).
    with open(args.meta_out, 'wb') as f:
        pickle.dump(np.array(metadata, dtype=np.int32), f, protocol=pickle.HIGHEST_PROTOCOL)

    # Write data atomically so an interrupt can't corrupt the source file
    # (data_out may equal src).
    tmp_out = args.data_out + ".tmp"
    with open(tmp_out, 'wb') as f:
        pickle.dump([tuple(pair) for pair in tokenized_dataset], f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_out, args.data_out)

    print(f"wrote {len(tokenized_dataset)} pairs -> {args.data_out}, {args.meta_out}")

if __name__ == "__main__":
    main()
