from flask import Flask, render_template, request

import joblib

import numpy as np

app = Flask(__name__)


clf_model = joblib.load("/home/astralfate/mysite/decision_tree_pipeline.pkl")


@app.route('/')

def home():

    return render_template('index.html')

@app.route('/result', methods=['POST'])

def result():

    # if request.method == 'POST':

    features = [float(x) for x in request.form.values()]

        # features = np.array(features).reshape(1, -1)
    features = [np.array(features)]

        # print(features)

    clf_pred = clf_model.predict(features)

        # print('The Classification Prediction is: ',clf_pred)



    return render_template('result.html',clf_pred=clf_pred[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

