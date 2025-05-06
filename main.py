from flask import Flask, request, jsonify, render_template
from feature_extractor import Feature_Extraction_img
import joblib
import os
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './temp'

svm_model = joblib.load('train/model.pkl')

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            if "image" not in request.files or request.files["image"].filename == "":
                return render_template("index.html", error="Chưa chọn ảnh")

            img_file = request.files["image"]
            img_filename = secure_filename(img_file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            img_file.save(temp_path)

            features = Feature_Extraction_img(temp_path)
            prediction = svm_model.predict(features)
            os.remove(temp_path)
            return render_template("index.html", prediction=int(prediction[0]))
        except Exception as e:
            return render_template("index.html", error="Lỗi hệ thống: " + str(e))
    
    return render_template("index.html", prediction=None)

# API endpoint (giữ nguyên nếu bạn vẫn cần POST API)
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        img_file = request.files["image"]
        if img_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        temp_dir = "./temp"
        img_filename = secure_filename(img_file.filename)
        temp_file_path = os.path.join(temp_dir, img_filename)
        img_file.save(temp_file_path)

        features = Feature_Extraction_img(temp_file_path)
        prediction = svm_model.predict(features)
        os.remove(temp_file_path)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
