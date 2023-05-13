from flask import Flask, request, jsonify
from flask_cors import CORS

from model import predict

import warnings

warnings.filterwarnings(
    action="ignore",
    message=".*Unverified HTTPS.*",
)

app = Flask(__name__)
CORS(app)


@app.route("/api/predict", methods=["POST"])
def api():
    data = request.get_json()

    if "initial" not in data:
        return jsonify({"error": "Initial text is required"}), 400
    if "nWords" not in data:
        return jsonify({"error": "Number of words is required"}), 400
    """
    Sample JSON input:
    {
        "initial": "I really like the Arctic Monkeys and ",
        "nWords": 100
    }
    """

    content = predict(data["initial"], data["nWords"])

    return jsonify({"content": content}), 200


if __name__ == "__main__":
    app.run(debug=True)
