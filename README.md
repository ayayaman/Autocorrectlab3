# NLP Lab 03 — Streamlit N-Gram Auto-Complete

This is the **deployment-ready GUI version** of the Statistical Language Modeling / N-gram Auto-Complete lab.

It follows the lab setup:

- Twitter corpus: `en_US.twitter.txt`
- 80% training / 20% testing
- `random.seed(87)`
- minimum word frequency = 2
- 1-gram through 5-gram models
- `<s>` start token and `<e>` end token
- `<unk>` for out-of-vocabulary words
- Add-k (Laplace) smoothing
- next-word prediction
- optional prefix filtering

## Files

Keep **all of these files in the root of the GitHub repository**:

```text
app.py
model_utils.py
train_model.py
requirements.txt
README.md
en_US.twitter.txt
autocomplete_model.pkl
```

### Important

`en_US.twitter.txt` is the **training dataset**.

`autocomplete_model.pkl` is the **already-trained model** created from that dataset. The Streamlit app loads the `.pkl` file; it does **not** retrain the model every time the web app starts.

The supplied project already contains both files, so you can deploy it directly.

## Run locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

To retrain the model from the Twitter dataset:

```bash
python train_model.py
```

This creates/updates:

```text
autocomplete_model.pkl
```

Then run:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create/open your GitHub repository.
2. Upload the **contents of this folder** to the repository root.
3. Make sure `app.py` is visible directly in the repository root.
4. Go to Streamlit Community Cloud and create/deploy the app.
5. Select your GitHub repository and branch `main`.
6. Set **Main file path** to:

```text
app.py
```

Do **not** use `streamlit_app.py` as the entrypoint.

The repository should look like:

```text
Autocorrectlab3/
├── app.py
├── model_utils.py
├── train_model.py
├── requirements.txt
├── README.md
├── en_US.twitter.txt
└── autocomplete_model.pkl
```

## GUI features

- Text input
- Suggest Next Word button
- Maximum N-gram order: 2–5
- Number of suggestions: 3–10
- Add-k smoothing value
- Optional prefix filtering
- Vocabulary/model statistics
- NLP prediction pipeline
- Ranked next-word suggestions with probabilities
- Comparison of different N-gram orders
- Tokenized input viewer

## Example

Try:

```text
I want to
```

Then click **Suggest Next Word**.

You can also use a prefix such as:

```text
c
```

to show only candidate words beginning with `c`.

## Auto-complete vs. spelling correction

This project implements **next-word auto-completion** using statistical N-gram language modeling.

It is not a full spelling-correction system. For example, changing `I lik` into `I like` would require an additional spelling-correction component.
