from flask import Flask, request, jsonify
import pandas as pd
import joblib

# ✅ Create Flask app
app = Flask(__name__)

# ✅ Load saved artifacts
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ✅ Load training columns
X_columns = joblib.load("X_columns.pkl")  # list of feature names used during training

# Numerical columns used during scaling
numerical_features = ['Soil_Moisture', 'Humidity', 'Temperature']


# ✅ Define endpoint safely (prevents duplicate definition in notebooks)
if '/predict_svm' not in [rule.rule for rule in app.url_map.iter_rules()]:

    @app.route('/predict_svm', methods=['POST'])
    def predict_svm():
        try:
            # 🔹 Get JSON input
            input_data = request.get_json(force=True)

            # 🔹 Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # 🔹 Reindex to match training columns
            processed_input = input_df.reindex(columns=X_columns, fill_value=0)

            # 🔹 Scale numerical features
            processed_input[numerical_features] = scaler.transform(
                processed_input[numerical_features]
            )

            # 🔹 Prediction
            prediction_encoded = svm_model.predict(processed_input)

            # 🔹 Decode label
            predicted_fertilizer = label_encoder.inverse_transform(prediction_encoded)

            return jsonify({
                'predicted_fertilizer': predicted_fertilizer[0]
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 400

else:
    print("Endpoint '/predict_svm' already exists.")
    

# ✅ Run server
if __name__ == '__main__':
    app.run(debug=True)
