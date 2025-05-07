from flask import Flask, request, jsonify, render_template
from feature_extractor import Feature_Extraction_img
import joblib
from io import BytesIO
from PIL import Image

app = Flask(__name__)
svm_model = joblib.load('train/model.pkl')

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            img_file = request.files["image"]
            if img_file.filename == "":
                return render_template("index.html", error="Chưa chọn ảnh")

            # Đọc ảnh trực tiếp từ bộ nhớ
            img = Image.open(BytesIO(img_file.read()))
            features = Feature_Extraction_img(img)
            prediction = svm_model.predict(features)
            label = 'Cat' if prediction[0] == 0 else 'Dog'

            return render_template("index.html", prediction=label)
        except Exception as e:
            return render_template("index.html", error="Lỗi hệ thống: " + str(e))

    return render_template("index.html", prediction=None)

# API endpoint không dùng file temp nữa
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        img_file = request.files["image"]
        if img_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Đọc ảnh trực tiếp từ bộ nhớ
        img = Image.open(BytesIO(img_file.read()))
        features = Feature_Extraction_img(img)
        prediction = svm_model.predict(features)
        label = 'Cat' if prediction[0] == 0 else 'Dog'

        return render_template("index.html", prediction=label)
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
