import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "StudentsPerformance.csv")
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_student_performance.joblib")


def load_data_and_encoders():
    """
    Muat dataset dan buat LabelEncoder untuk setiap kolom kategorikal.
    Proses ini meniru persis langkah di notebook:
      - hitung average_score
      - buat kolom pass (target)
      - X = df.drop(columns=['pass'])
      - label encoding untuk kolom kategorikal di X
    Hasilnya: dict encoders per kolom kategorikal.
    """
    df = pd.read_csv(CSV_PATH)

    # Buat kolom rata-rata dan target (pass) sesuai notebook
    df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
    df["pass"] = df["average_score"].apply(lambda x: 1 if x >= 70 else 0)

    X = df.drop(columns=["pass"])

    label_cols = X.select_dtypes(include="object").columns
    encoders = {}

    for col in label_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    feature_order = list(X.columns)
    return encoders, feature_order


def load_model():
    return joblib.load(MODEL_PATH)


encoders, feature_order = load_data_and_encoders()
model = load_model()

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            # Ambil input dari form
            gender = request.form.get("gender")
            race_ethnicity = request.form.get("race_ethnicity")
            parental_edu = request.form.get("parental_edu")
            lunch = request.form.get("lunch")
            prep_course = request.form.get("prep_course")

            math_score = float(request.form.get("math_score"))
            reading_score = float(request.form.get("reading_score"))
            writing_score = float(request.form.get("writing_score"))

            # Buat DataFrame satu baris sesuai dengan kolom asli
            data_dict = {
                "gender": gender,
                "race/ethnicity": race_ethnicity,
                "parental level of education": parental_edu,
                "lunch": lunch,
                "test preparation course": prep_course,
                "math score": math_score,
                "reading score": reading_score,
                "writing score": writing_score,
            }

            # Hitung average_score seperti di notebook
            avg_score = (math_score + reading_score + writing_score) / 3.0
            data_dict["average_score"] = avg_score

            input_df = pd.DataFrame([data_dict])

            # Label encoding untuk kolom kategorikal menggunakan encoder yang sudah dilatih
            for col, le in encoders.items():
                if col in input_df.columns:
                    # Jika nilai baru tidak pernah muncul di training, tangani sederhana:
                    # gunakan kategori pertama encoder (menghindari error).
                    if input_df[col].iloc[0] not in le.classes_:
                        # extend classes_ agar transform tidak error
                        le_classes = list(le.classes_)
                        le_classes.append(input_df[col].iloc[0])
                        le.classes_ = np.array(le_classes)

                    input_df[col] = le.transform(input_df[col])

            # Pastikan urutan fitur sama dengan saat training
            input_df = input_df[feature_order]

            y_pred = model.predict(input_df)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                # probabilitas kelas 1 (lulus)
                probability = float(proba[1]) * 100.0

            prediction = "Lulus" if y_pred == 1 else "Tidak Lulus"

        except Exception as e:
            error = f"Terjadi kesalahan saat memproses input: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
    )


if __name__ == "__main__":
    # Debug bisa diubah ke False saat production
    app.run(host="0.0.0.0", port=5000, debug=True)


