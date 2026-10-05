"""
Utilities for the NLP Lab 3 Statistical Language Modeling Streamlit app.

The preprocessing and N-gram counting follow the uploaded Lab 3 notebook:
- 80/20 train/test split
- random.seed(87)
- minimum word frequency = 2
- N-gram orders 1 through 5
- <s> start token and <e> end token
"""

import re
from collections import Counter
from typing import Dict, List, Tuple, Optional


def tokenize_sentences(sentences: List[str]) -> List[List[str]]:
    """Tokenize sentences similarly to nltk.word_tokenize and lowercase them."""
    # The lab uses nltk.word_tokenize. This regex keeps common punctuation as
    # separate tokens and avoids requiring NLTK at runtime in the Streamlit app.
    tokenized = []
    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    for sentence in sentences:
        sentence = sentence.lower()
        tokenized.append(pattern.findall(sentence))

    return tokenized


def split_to_sentences(data: str) -> List[str]:
    sentences = data.split("\n")
    sentences = [s.strip() for s in sentences]
    return [s for s in sentences if s]


def get_tokenized_data(data: str) -> List[List[str]]:
    return tokenize_sentences(split_to_sentences(data))


def count_words(tokenized_sentences: List[List[str]]) -> Dict[str, int]:
    counts = Counter()
    for sentence in tokenized_sentences:
        counts.update(sentence)
    return dict(counts)


def get_words_with_nplus_frequency(
    tokenized_sentences: List[List[str]],
    count_threshold: int
) -> List[str]:
    word_counts = count_words(tokenized_sentences)
    return [word for word, cnt in word_counts.items() if cnt >= count_threshold]


def replace_oov_words_by_unk(
    tokenized_sentences: List[List[str]],
    vocabulary: List[str],
    unknown_token: str = "<unk>"
) -> List[List[str]]:
    vocabulary = set(vocabulary)
    return [
        [token if token in vocabulary else unknown_token for token in sentence]
        for sentence in tokenized_sentences
    ]


def preprocess_data(
    train_data: List[List[str]],
    test_data: List[List[str]],
    count_threshold: int
):
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary)
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary)
    return train_data_replaced, test_data_replaced, vocabulary


def count_n_grams(
    data: List[List[str]],
    n: int,
    start_token: str = "<s>",
    end_token: str = "<e>"
) -> Dict[Tuple[str, ...], int]:
    """Match the notebook's count_n_grams implementation."""
    n_grams = {}

    for sentence in data:
        sentence = [start_token] * n + sentence + [end_token]
        sentence = tuple(sentence)

        for i in range(len(sentence) - n + 1):
            n_gram = sentence[i:i + n]
            n_grams[n_gram] = n_grams.get(n_gram, 0) + 1

    return n_grams


def estimate_probability(
    word: str,
    previous_n_gram: Tuple[str, ...],
    n_gram_counts: Dict[Tuple[str, ...], int],
    n_plus1_gram_counts: Dict[Tuple[str, ...], int],
    vocabulary_size: int,
    k: float = 1.0
) -> float:
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    denominator = previous_n_gram_count + k * vocabulary_size

    n_plus1_gram = previous_n_gram + (word,)
    numerator = n_plus1_gram_counts.get(n_plus1_gram, 0) + k

    return numerator / denominator if denominator else 0.0


def rank_next_words(
    previous_tokens: List[str],
    n_gram_counts: Dict[Tuple[str, ...], int],
    n_plus1_gram_counts: Dict[Tuple[str, ...], int],
    vocabulary: List[str],
    k: float = 1.0,
    start_with: Optional[str] = None,
    top_k: int = 5
):
    """
    Rank candidate next words using the same additive-smoothed probability
    used in the notebook's suggest_a_word().
    """
    if not n_gram_counts:
        return []

    n = len(next(iter(n_gram_counts)))
    previous_n_gram = tuple(previous_tokens[-n:])

    # The notebook adds <e> and <unk> to the candidate vocabulary.
    candidates = list(vocabulary)
    if "<e>" not in candidates:
        candidates.append("<e>")
    if "<unk>" not in candidates:
        candidates.append("<unk>")

    vocabulary_size = len(candidates)
    results = []

    for word in candidates:
        if start_with and not word.startswith(start_with.lower()):
            continue

        probability = estimate_probability(
            word,
            previous_n_gram,
            n_gram_counts,
            n_plus1_gram_counts,
            vocabulary_size,
            k=k
        )
        results.append((word, probability))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def get_suggestions_for_orders(
    previous_tokens: List[str],
    n_gram_counts_list: List[Dict[Tuple[str, ...], int]],
    vocabulary: List[str],
    k: float = 1.0,
    top_k: int = 5,
    max_order: Optional[int] = None,
    start_with: Optional[str] = None
):
    """
    Return ranked suggestions for bigram through the selected maximum order.

    The notebook's get_suggestions() compares adjacent N-gram models:
    unigram -> bigram, bigram -> trigram, etc.
    """
    if max_order is None:
        max_order = len(n_gram_counts_list)

    max_order = max(2, min(max_order, len(n_gram_counts_list)))

    all_results = []

    for i in range(max_order - 1):
        n = i + 2
        results = rank_next_words(
            previous_tokens,
            n_gram_counts_list[i],
            n_gram_counts_list[i + 1],
            vocabulary,
            k=k,
            start_with=start_with,
            top_k=top_k
        )
        all_results.append({
            "order": n,
            "label": f"{n}-gram",
            "results": results
        })

    return all_results
