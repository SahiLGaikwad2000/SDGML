from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('levl.pki', 'rb'))
app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['intern']
    data2 = request.form['proj']
    data3 = request.form['cgpa']
    arr = np.array([[data1, data2, data3]])
    predi = model.predict(arr)
    return render_template('after.html', data=predi)


if __name__ == "__main__":
    app.run(debug=True)
