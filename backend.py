import pickle
from flask import Flask, request, jsonify
import xgboost as xgb

app = Flask('ping')

with open('model.bin', 'rb') as f_in: 
    dv, model = pickle.load(f_in)


client = {"age": 35,
 "gender": "Male",
 "weight_(kg)": 102.5,
 "height_(m)": 1.94,
 "max_bpm": 183,
 "avg_bpm": 158,
 "resting_bpm": 64,
 "session_duration_(hours)": 0.84,
 "workout_type": "Cardio",
 "fat_percentage": 21.1,
 "water_intake_(liters)": 2.4,
 "workout_frequency_(days/week)": 2,
 "experience_level": 1,
 "bmi": 27.23}


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    dnew = xgb.DMatrix(X, feature_names=dv.get_feature_names_out().tolist())
    y_pred = model.predict(dnew)

    result = {
        'burned_calories_per_session': float(y_pred.round(2))
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)