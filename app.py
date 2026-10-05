import os
import pickle
import re
import streamlit as st

from model_utils import get_suggestions_for_orders


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="N-Gram Auto-Complete",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        background-color: #f7f9fc;
    }

    /* Header */
    .app-header {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #eef2ff, #f8fafc);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.2rem;
    }

    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #172033;
    }

    .app-subtitle {
        font-size: 1rem;
        color: #64748b;
    }

    /* Cards */
    .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 750;
        color: #172033;
        margin-bottom: 0.65rem;
    }

    /* Suggestion cards */
    .suggestion {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.75rem 1rem;
        margin: 0.45rem 0;
    }

    .suggestion-word {
        font-size: 1.1rem;
        font-weight: 750;
        color: #111827;
    }

    .suggestion-rank {
        color: #64748b;
        font-size: 0.85rem;
    }

    .probability {
        font-weight: 700;
        color: #475569;
    }

    /* Pipeline */
    .pipeline {
        display: flex;
        gap: 0.35rem;
        flex-wrap: wrap;
        align-items: center;
        margin-top: 0.5rem;
    }

    .pipeline-item {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.45rem 0.7rem;
        font-size: 0.84rem;
        color: #334155;
    }

    .arrow {
        color: #94a3b8;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    /* Button */
    div.stButton > button {
        border-radius: 11px;
        min-height: 2.8rem;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_FILE = "autocomplete_model.pkl"


@st.cache_resource(show_spinner=False)
def load_model():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    return model


# ============================================================
# TOKENIZER FOR USER INPUT
# ============================================================

def tokenize_input(text):
    text = text.lower().strip()

    # Similar tokenization behavior to the lab.
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 NLP Model")

    st.markdown(
        """
        **Statistical Language Modeling**

        This application uses N-gram probabilities
        to predict the next word.
        """
    )

    st.divider()

    st.markdown("### ⚙️ Model Settings")

    max_order = st.slider(
        "Maximum N-gram order",
        min_value=2,
        max_value=5,
        value=5,
        help="Higher values use more previous context."
    )

    top_k = st.slider(
        "Number of suggestions",
        min_value=3,
        max_value=10,
        value=5
    )

    smoothing_k = st.number_input(
        "Smoothing k",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    prefix = st.text_input(
        "Optional word prefix",
        placeholder="e.g. c"
    )

    st.divider()

    st.markdown("### 📚 N-gram Models")

    st.markdown(
        """
        - Unigram
        - Bigram
        - Trigram
        - 4-gram
        - 5-gram
        """
    )

    st.divider()

    st.caption("NLP Lab 03 — Statistical Language Modeling")


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):

    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">📝 N-Gram Auto-Complete</div>
            <div class="app-subtitle">
                Statistical Language Modeling Interface
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.error("Model file not found.")

    st.markdown(
        """
        ### First run

        Put these files in the same folder:

        ```
        app.py
        model_utils.py
        train_model.py
        en_US.twitter.txt
        ```

        Then run:

        ```bash
        python train_model.py
        ```

        After the model is created:

        ```bash
        streamlit run app.py
        ```
        """
    )

    st.stop()


model = load_model()

n_gram_counts_list = model["n_gram_counts_list"]
vocabulary = model["vocabulary"]
config = model.get("config", {})


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">📝 N-Gram Auto-Complete</div>
        <div class="app-subtitle">
            Predict the most likely next word using a statistical
            language model trained on Twitter text.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns([1.55, 1], gap="large")


# ============================================================
# INPUT
# ============================================================

with left:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">✍️ Enter your text</div>',
        unsafe_allow_html=True
    )

    text = st.text_area(
        "Movie review / sentence / text",
        placeholder="Example: I want to go",
        height=155,
        label_visibility="collapsed",
    )

    button = st.button(
        "🔮 Suggest Next Word",
        type="primary",
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MODEL INFORMATION
# ============================================================

with right:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">📊 Model Status</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Vocabulary",
            f"{len(vocabulary):,}"
        )

    with c2:
        st.metric(
            "Max Order",
            f"{len(n_gram_counts_list)}"
        )

    st.caption(
        f"Minimum frequency: "
        f"{config.get('minimum_freq', 2)}"
    )

    st.caption(
        f"Training split: "
        f"{config.get('train_ratio', 0.80):.0%}"
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PIPELINE
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">🧩 NLP Prediction Pipeline</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="pipeline">
        <span class="pipeline-item">User Text</span>
        <span class="arrow">→</span>
        <span class="pipeline-item">Tokens</span>
        <span class="arrow">→</span>
        <span class="pipeline-item">Previous N-gram</span>
        <span class="arrow">→</span>
        <span class="pipeline-item">Candidate Words</span>
        <span class="arrow">→</span>
        <span class="pipeline-item">Probabilities</span>
        <span class="arrow">→</span>
        <span class="pipeline-item">Ranking</span>
        <span class="arrow">→</span>
        <span class="pipeline-item">Suggestion</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PREDICTION
# ============================================================

if button:

    if not text.strip():

        st.warning("Please enter some text first.")

    else:

        tokens = tokenize_input(text)

        if not tokens:

            st.warning("No valid tokens were found.")

        else:

            with st.spinner("Calculating N-gram probabilities..."):

                prefix_value = prefix.strip().lower() or None

                all_results = get_suggestions_for_orders(
                    previous_tokens=tokens,
                    n_gram_counts_list=n_gram_counts_list,
                    vocabulary=vocabulary,
                    k=smoothing_k,
                    top_k=top_k,
                    max_order=max_order,
                    start_with=prefix_value,
                )

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-title">💡 Suggested Next Words</div>',
                unsafe_allow_html=True
            )

            if prefix_value:
                st.caption(
                    f"Showing words starting with: `{prefix_value}`"
                )

            # Display highest-order model first as the primary result.
            primary = all_results[-1] if all_results else None

            if primary and primary["results"]:

                st.markdown(
                    f"#### ⭐ {primary['label']} Prediction"
                )

                for rank, (word, probability) in enumerate(
                    primary["results"],
                    start=1
                ):

                    display_word = (
                        "End of sentence"
                        if word == "<e>"
                        else word
                    )

                    st.markdown(
                        f"""
                        <div class="suggestion">
                            <span class="suggestion-rank">
                                #{rank}
                            </span>
                            &nbsp;&nbsp;
                            <span class="suggestion-word">
                                {display_word}
                            </span>
                            <span style="float:right"
                                  class="probability">
                                {probability:.2%}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            else:

                st.info(
                    "No candidate matched the selected context/prefix."
                )

            st.markdown("</div>", unsafe_allow_html=True)


            # ------------------------------------------------
            # Compare N-gram orders
            # ------------------------------------------------

            if len(all_results) > 1:

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="section-title">🔬 Compare N-gram Orders</div>',
                    unsafe_allow_html=True
                )

                for item in all_results:

                    results = item["results"]

                    if not results:
                        continue

                    best_word, best_prob = results[0]

                    display_word = (
                        "End of sentence"
                        if best_word == "<e>"
                        else best_word
                    )

                    st.write(
                        f"**{item['label']}** → "
                        f"`{display_word}` "
                        f"({best_prob:.2%})"
                    )

                st.markdown("</div>", unsafe_allow_html=True)


            # ------------------------------------------------
            # Token information
            # ------------------------------------------------

            with st.expander("🔍 View tokenized input"):

                st.write(tokens)

                st.caption(
                    f"Number of tokens: {len(tokens)}"
                )


# ============================================================
# EDUCATIONAL INFORMATION
# ============================================================

with st.expander("📘 How does this work?"):

    st.markdown(
        """
        ### Statistical Auto-Complete

        The system uses the words already typed as context and
        estimates the probability of possible next words.

        For example:

        **Input**

        `I like`

        **Prediction**

        `a`

        The probability is calculated using the N-gram counts
        learned from the training corpus.

        ### Add-k smoothing

        The model uses:

        **P(w | h) = (Count(h,w) + k) / (Count(h) + kV)**

        This prevents unseen N-grams from receiving zero probability.

        ### Why use higher-order N-grams?

        A bigram uses one previous word.

        A trigram uses two previous words.

        A 4-gram uses three previous words.

        A 5-gram uses four previous words.

        Higher-order models use more context, but they can suffer
        from data sparsity.
        """
    )


st.divider()

st.caption(
    "NLP Lab 03 • Statistical Language Modeling • "
    "N-gram Auto-Complete"
)
