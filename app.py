from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("model/logistic_regression_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

label_decoder = {
    0: "Non-KBGO",
    1: "KBGO"
}

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    x = vectorizer.transform([text])
    prediction = model.predict(x)[0]
    return jsonify({"result": label_decoder[prediction]})

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
