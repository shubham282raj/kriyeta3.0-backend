from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.sentiment_analysis_roberta import sentiment_analysis

app = Flask(__name__)
CORS(app,origins=["*"], supports_credentials=True)

@app.route("/api/sentiment", methods = ["POST"])
def sentiment():
    req = request.get_json()
    print(req)
    req = sentiment_analysis(req['prompt'])
    print(req)
    return jsonify(req)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)