# ============================================================
# WEEK 4 - AI PROJECT DEPLOYMENT
# Flask Prediction Application
# ============================================================

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np


# ------------------------------------------------------------
# 1. CREATE FLASK APPLICATION
# ------------------------------------------------------------

app = Flask(__name__)


# ------------------------------------------------------------
# 2. LOAD SAVED MODEL
# ------------------------------------------------------------

model_data = joblib.load(
    "model/iris_model.joblib"
)

model = model_data["model"]

target_names = model_data["target_names"]

feature_names = model_data["feature_names"]


# ------------------------------------------------------------
# 3. HOME PAGE
# ------------------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ------------------------------------------------------------
# 4. PREDICTION API
# ------------------------------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # Get JSON data

        data = request.get_json()

        # Extract input values

        sepal_length = float(
            data["sepal_length"]
        )

        sepal_width = float(
            data["sepal_width"]
        )

        petal_length = float(
            data["petal_length"]
        )

        petal_width = float(
            data["petal_width"]
        )


        # Create feature array

        features = np.array([
            [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        ])


        # Make prediction

        prediction = model.predict(
            features
        )[0]


        # Get prediction probabilities

        probabilities = model.predict_proba(
            features
        )[0]


        # Get confidence

        confidence = (
            probabilities[prediction] * 100
        )


        # Get flower name

        flower_name = target_names[
            prediction
        ]


        # Return response

        return jsonify({

            "success": True,

            "prediction": flower_name,

            "confidence": round(
                confidence,
                2
            ),

            "probabilities": {

                target_names[i]:
                round(
                    probabilities[i] * 100,
                    2
                )

                for i in range(
                    len(target_names)
                )
            }

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 400


# ------------------------------------------------------------
# 5. HEALTH CHECK API
# ------------------------------------------------------------

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "healthy",

        "model": "Iris Logistic Regression",

        "api": "running"

    })


# ------------------------------------------------------------
# 6. RUN APPLICATION
# ------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )