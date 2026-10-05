# NLP Lab 03 — Streamlit N-Gram Auto-Complete

This folder adds a graphical Streamlit interface to the Statistical
Language Modeling / N-gram Auto-Complete lab.

The implementation follows the uploaded lab notebook:

- Twitter corpus: `en_US.twitter.txt`
- 80% training / 20% testing
- `random.seed(87)`
- minimum word frequency = 2
- N-gram models from 1-gram through 5-gram
- `<s>` start token
- `<e>` end token
- `<unk>` for low-frequency / OOV words
- additive (Laplace / Add-k) smoothing
- next-word prediction
- optional prefix filtering

## Files

```text
NLP_Lab3_Streamlit_AutoComplete/
│
├── app.py
├── model_utils.py
├── train_model.py
├── requirements.txt
├── README.md
└── en_US.twitter.txt       <-- copy your dataset here
```

## Step 1 — Install Streamlit

Open Anaconda Prompt / Terminal in this folder:

```bash
pip install -r requirements.txt
```

## Step 2 — Put the dataset here

Copy:

```text
en_US.twitter.txt
```

into the same folder as `train_model.py`.

## Step 3 — Build the model

Run:

```bash
python train_model.py
```

This creates:

```text
autocomplete_model.pkl
```

Training can take some time because the lab creates five N-gram
count dictionaries.

## Step 4 — Start Streamlit

Run:

```bash
streamlit run app.py
```

The browser will open the application.

Usually:

```text
http://localhost:8501
```

## Important

Do NOT train the model every time Streamlit reloads.

The model is trained once by:

```bash
python train_model.py
```

Then `app.py` loads the saved `autocomplete_model.pkl`.

## Interface features

The GUI contains:

- text input
- Suggest Next Word button
- number of suggestions
- maximum N-gram order
- smoothing k
- optional prefix filter
- vocabulary size
- model information
- prediction pipeline
- ranked suggestions
- probability percentages
- comparison of N-gram orders
- tokenized input viewer
- explanation of the statistical model

## Example

Enter:

```text
i am to
```

The model will calculate candidate probabilities and display the
highest-ranked next words.

You can also enter a prefix:

```text
d
```

to restrict suggestions to words beginning with `d`.

## Relation to the lab

The lab's `suggest_a_word()` function predicts the most likely next
word from the vocabulary and supports an optional `start_with` prefix.
The Streamlit interface exposes that same idea as an interactive GUI.

This application is an auto-complete interface. It is NOT a full
spelling-correction system. For example, correcting:

```text
I lik
```

to:

```text
I like
```

would require an additional spelling-correction component.
