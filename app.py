"""
FAQ Chatbot - Web UI (Flask)
============================
Run with: python app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify
from chatbot import FAQChatbot

app = Flask(__name__)
bot = FAQChatbot("faqs.csv")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json.get("message", "")
    answer, matched_question, score = bot.get_response(user_message)
    return jsonify({
        "answer": answer,
        "matched_question": matched_question,
        "confidence": round(float(score), 2)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
