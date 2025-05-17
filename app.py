from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained sentiment model
with open('models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.json.get('review', '')
    if not review:
        return jsonify({'error': 'No input provided'}), 400
    
    prediction = model.predict([review])[0]
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(debug=True)
