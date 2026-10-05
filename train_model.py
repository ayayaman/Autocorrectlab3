"""
Train and save the Statistical Language Model used by the Streamlit app.

Place this file in the same folder as:
    en_US.twitter.txt

Run:
    python train_model.py

The training configuration follows the uploaded NLP Lab 3 notebook:
    random.seed(87)
    train_size = 80%
    minimum_freq = 2
    N-gram orders = 1..5
"""

import os
import pickle
import random
import time

from model_utils import (
    get_tokenized_data,
    preprocess_data,
    count_n_grams,
)

DATA_FILE = "en_US.twitter.txt"
MODEL_FILE = "autocomplete_model.pkl"

RANDOM_SEED = 87
TRAIN_RATIO = 0.80
MINIMUM_FREQ = 2
MAX_N = 5


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Could not find '{DATA_FILE}'. "
            "Put the Twitter dataset in the same folder as train_model.py."
        )

    start = time.time()

    print("=" * 60)
    print("NLP Lab 3 - Statistical Language Model Training")
    print("=" * 60)

    print("\n[1/5] Loading Twitter data...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = f.read()

    print(f"Characters loaded: {len(data):,}")

    print("\n[2/5] Tokenizing...")
    tokenized_data = get_tokenized_data(data)
    print(f"Sentences: {len(tokenized_data):,}")

    random.seed(RANDOM_SEED)
    random.shuffle(tokenized_data)

    train_size = int(len(tokenized_data) * TRAIN_RATIO)
    train_data = tokenized_data[:train_size]
    test_data = tokenized_data[train_size:]

    print(f"Training sentences: {len(train_data):,}")
    print(f"Testing sentences:  {len(test_data):,}")

    print("\n[3/5] Handling low-frequency words...")
    train_data_processed, test_data_processed, vocabulary = preprocess_data(
        train_data,
        test_data,
        MINIMUM_FREQ
    )

    print(f"Minimum frequency: {MINIMUM_FREQ}")
    print(f"Vocabulary size: {len(vocabulary):,}")

    print("\n[4/5] Building N-gram counts...")

    n_gram_counts_list = []

    for n in range(1, MAX_N + 1):
        t0 = time.time()

        print(f"  Computing {n}-gram counts...", end=" ", flush=True)

        counts = count_n_grams(train_data_processed, n)
        n_gram_counts_list.append(counts)

        print(
            f"{len(counts):,} unique N-grams "
            f"({time.time() - t0:.1f}s)"
        )

    print("\n[5/5] Saving model...")

    model = {
        "n_gram_counts_list": n_gram_counts_list,
        "vocabulary": vocabulary,
        "config": {
            "random_seed": RANDOM_SEED,
            "train_ratio": TRAIN_RATIO,
            "minimum_freq": MINIMUM_FREQ,
            "max_n": MAX_N,
            "dataset": DATA_FILE,
        },
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(MODEL_FILE) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("MODEL READY")
    print("=" * 60)
    print(f"Saved as: {MODEL_FILE}")
    print(f"Model size: {size_mb:.1f} MB")
    print(f"Total time: {time.time() - start:.1f}s")
    print("\nNow run:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()
