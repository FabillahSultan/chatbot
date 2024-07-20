from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import chatbot_response, intents, model, words, lemmatizer

app = Flask(__name__)
CORS(app)  # Aktifkan CORS untuk semua rute
app.static_folder = 'static'

@app.route("/")
def home():
    return jsonify({'api success': True})

@app.route("/get", methods=["POST"])
def get_bot_response():
    userText = request.json.get('msg')
    response = chatbot_response(userText, intents, model, words, lemmatizer)
    return jsonify({'answer': response})

if __name__ == "__main__":
    app.run(debug=True)
