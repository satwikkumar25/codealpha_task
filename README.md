# FAQ Chatbot (Machine Learning / NLP Project)

A chatbot that answers frequently asked questions by matching a user's
query against a FAQ dataset using **NLP preprocessing (NLTK)** and
**TF-IDF + cosine similarity**.

## Project Structure

```
faq_chatbot/
├── faqs.csv           # FAQ dataset (question, answer pairs) — 20 sample FAQs
├── chatbot.py          # Core engine: preprocessing + matching + CLI
├── app.py               # Optional Flask web UI
├── templates/
│   └── index.html       # Chat interface (HTML/CSS/JS)
└── README.md
```

## How it works (maps to the task checklist)

1. **Collect FAQs** — `faqs.csv` holds question/answer pairs. Replace this
   with your own topic/product FAQs (keep the same two-column format).
2. **Preprocess the text** — `preprocess()` in `chatbot.py` lowercases text,
   strips punctuation, tokenizes with NLTK, removes stopwords, and
   lemmatizes each token.
3. **Match user questions** — `FAQChatbot` fits a `TfidfVectorizer` on all
   preprocessed FAQ questions. A user query is vectorized the same way, and
   `cosine_similarity` scores it against every FAQ question. The highest
   score above a threshold (default `0.25`) is selected as the match.
4. **Display the best matching answer** — the corresponding answer is
   returned as the chatbot's response. Below the threshold, it returns a
   fallback ("couldn't find a matching answer") message.
5. **Optional chat UI** — `app.py` + `templates/index.html` provide a
   simple Flask-based web chat interface.

## Setup

```bash
pip install nltk scikit-learn pandas flask
```

The first run of `chatbot.py` automatically downloads the required NLTK
data (`punkt`, `stopwords`, `wordnet`).

## Usage

### Option A: Command-line chatbot
```bash
python chatbot.py
```
Type a question, get an answer. Type `quit` or `exit` to stop.

### Option B: Web chat UI
```bash
python app.py
```
Then open **http://127.0.0.1:5000** in your browser.

## Customizing

- **Add your own FAQs**: edit `faqs.csv` — just keep the `question,answer`
  columns (wrap any text containing commas in double quotes).
- **Tune sensitivity**: change `similarity_threshold` when creating
  `FAQChatbot(csv_path, similarity_threshold=0.25)` — lower it to match more
  loosely-worded questions, raise it to be stricter.
- **Debugging matches**: use `bot.top_matches(query, top_n=3)` to see the
  top 3 candidate FAQs and their similarity scores for any query.

## Notes / Possible Extensions

- TF-IDF + cosine similarity is a classic **keyword-overlap** method: it
  matches well when the user's wording shares terms with the FAQ, but won't
  catch pure synonyms (e.g., "money back" vs "refund") unless you expand
  vocabulary or switch to embeddings.
- For a stronger model, you could replace TF-IDF with sentence embeddings
  (e.g., `sentence-transformers`) and cosine similarity on embedding
  vectors — this captures semantic meaning, not just shared words.
- You could also add **intent classification** (train a classifier on
  FAQ categories) as an alternative/complementary matching technique.
