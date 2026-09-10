# Create a base Flask server

import csv
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Enable cors
@app.after_request
def after_request(response):
    """
    Enable CORS
    """
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


# Load model and airport data once when the process starts. Keeping the
# relatively static airport catalog out of the request path avoids disk I/O
# and CSV parsing for every /airports request.
model = pickle.load(open('model.pkl', 'rb'))

with open('airports.csv', newline='', encoding='utf-8') as airport_file:
    AIRPORTS = [
        {'id': int(row['OriginAirportID']), 'name': row['OriginAirportName']}
        for row in csv.DictReader(airport_file)
    ]
AIRPORTS = sorted(AIRPORTS, key=lambda airport: airport['name'])

# Model takes two parameters - day of week and airport id, then returns a prediction of flight delay
@app.route('/predict', methods=['GET'])
def predict():
    """
    Takes two parameters - day of week and airport id, then returns a prediction of flight delay
    """
    # Store day_of_week as int
    day_of_week = int(request.args.get('day_of_week'))
    airport_id = int(request.args.get('airport_id'))
    prediction = model.predict_proba([[day_of_week, airport_id]])[0]
    
    # Split prediction string by space
    prediction = str(prediction).split(' ')

    # store first value from prediction as certainty, and remove the first character
    certainty = float(prediction[0][2:])

    # store second value from prediction as delay, and remove the last character
    delay = float(prediction[1][:-1])

    # return prediction as json
    return jsonify({'certainty': certainty, 'delay': delay})

# Create a new route called airports with method of get
@app.route('/airports', methods=['GET'])
def airports():
    return jsonify(AIRPORTS)

if __name__ == '__main__':
    app.run(debug=True)