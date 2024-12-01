from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form['features']
        np_data = np.array(data.split(','), dtype=np.float32).reshape(1, -1)

        if len(np_data[0]) != model.n_features_in_:
            return render_template("index.html", message="Error: Invalid number of features.")

        prediction = model.predict(np_data)
        result = "Parkinson's Disease Detected" if prediction == 1 else "No Parkinson's Disease"
        return render_template("index.html", message=result)
    except Exception as e:
        return render_template("index.html", message=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
