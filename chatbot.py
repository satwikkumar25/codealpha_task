"""
FAQ Chatbot - Core Engine
==========================
Steps implemented (matches the project task sheet):
1. Collect FAQs               -> loaded from faqs.csv
2. Preprocess text (NLTK)     -> lowercase, tokenize, remove stopwords, lemmatize
3. Match user question        -> TF-IDF vectorization + cosine similarity
4. Display best matching answer
"""

import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. One-time NLTK downloads (safe to re-run; will skip if already present)
# ---------------------------------------------------------------------------
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else
                        f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    """Lowercase, remove punctuation, tokenize, remove stopwords, lemmatize."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and tok.strip() != ""
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 3. Chatbot class: loads FAQs, builds TF-IDF matrix, matches queries
# ---------------------------------------------------------------------------
class FAQChatbot:
    def __init__(self, csv_path: str, similarity_threshold: float = 0.25):
        self.df = pd.read_csv(csv_path)
        self.similarity_threshold = similarity_threshold

        # Preprocess every FAQ question once, up front
        self.df["processed_question"] = self.df["question"].apply(preprocess)

        # Fit TF-IDF vectorizer on the cleaned FAQ questions
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["processed_question"]
        )

    def get_response(self, user_query: str):
        """
        Returns (answer, matched_question, similarity_score).
        If no FAQ is similar enough, returns a fallback message.
        """
        cleaned_query = preprocess(user_query)
        query_vec = self.vectorizer.transform([cleaned_query])

        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score < self.similarity_threshold:
            return (
                "Sorry, I couldn't find a matching answer. "
                "Could you rephrase your question or contact support@example.com?",
                None,
                best_score,
            )

        matched_question = self.df.iloc[best_idx]["question"]
        answer = self.df.iloc[best_idx]["answer"]
        return answer, matched_question, best_score

    def top_matches(self, user_query: str, top_n: int = 3):
        """Utility: return top-N candidate FAQs with scores (useful for debugging/demo)."""
        cleaned_query = preprocess(user_query)
        query_vec = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]
        return [
            (self.df.iloc[i]["question"], self.df.iloc[i]["answer"], similarities[i])
            for i in top_indices
        ]


# ---------------------------------------------------------------------------
# 4. CLI chat loop
# ---------------------------------------------------------------------------
def run_cli():
    print("=" * 60)
    print(" FAQ Chatbot (type 'quit' or 'exit' to stop)")
    print("=" * 60)

    bot = FAQChatbot("faqs.csv")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Bot: Goodbye! 👋")
            break
        if not user_input:
            continue

        answer, matched_q, score = bot.get_response(user_input)
        print(f"Bot: {answer}")
        # Uncomment below to see debug info (which FAQ matched + confidence)
        # print(f"   [matched: '{matched_q}' | score: {score:.2f}]")


if __name__ == "__main__":
    run_cli()
