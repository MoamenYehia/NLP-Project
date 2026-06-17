from flask import Flask, render_template, request, jsonify

from sentiment_engine import analyze

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_text():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Text too long — maximum 5000 characters."}), 400
    if len(text.split()) < 3:
        return jsonify({"error": "Please enter at least a few words."}), 400

    try:
        result = analyze(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
