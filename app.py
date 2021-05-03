from pymongo import MongoClient
from bson.objectid import ObjectId
from flask import Flask, jsonify, request, send_file, Response
from forecasting import testing, testingIndividual
from flask_cors import *
import os
import shutil
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_jwt_extended import create_access_token
import pandas as pd

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'Files')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = "bsajdknasjkbdjksabfjksbfjkdsankjs"

host = "mongodb+srv://admin:1234@cluster0.11ogg.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
db_name = 'Sales'
db = MongoClient(host)[db_name]

bcrypt = Bcrypt(app)
jwt = JWTManager(app)


@app.route('/')
def home():
    return "Sales Forecasting Prediction"


@app.route('/users/register', methods=["POST"])
def register():
    users = db.Users
    phone = request.get_json()['phone']
    response = users.find_one({'phone': phone})
    if (not response):
        firstName = request.get_json()['firstName']
        lastName = request.get_json()['lastName']
        phone = request.get_json()['phone']
        email = request.get_json()['email']
        password = request.get_json()['password']

        user_id = users.insert({
            'firstName': firstName,
            'lastName': lastName,
            'phone': phone,
            'email': email,
            'password': bcrypt.generate_password_hash(password).decode('utf-8')
        })

        new_user = users.find_one({'_id': user_id})

        result = {'phone': new_user['phone'] + ' registered'}

        return jsonify({'result': result})

    else:
        return jsonify({'result': "User already exists"})


@app.route('/users/login', methods=['POST'])
def login():
    users = db.Users

    phone = request.get_json()['phone']
    password = request.get_json()['password']
    result = ""

    response = users.find_one({'phone': phone})
    if response:
        if bcrypt.check_password_hash(response['password'], password):
            access_token = create_access_token(identity={
                'firstName': response['firstName'],
                'lastName': response['lastName'],
                'email': response['email']
            })
            result = jsonify({
                'name': response['firstName']
            })
        else:
            result = jsonify({"error": "Invalid username and password"})
    else:
        result = jsonify({"result": "No results found"})
    return result


@app.route('/download', methods=['POST'])
def downloadCSV():
    prediction = request.get_json()['prediction']
    file = request.get_json()['file']
    indata = pd.read_csv(f'Files/{file}')
    zipped = pd.DataFrame(zip(prediction))
    # axis=1 indicates to concat the frames column wise
    outdata = pd.concat([indata, zipped], axis=1)
    # we dont want headers and dont want the row labels
    outdata.to_csv(f'Files/prediction_{file}', header=['Day', 'Month', 'Year', 'Quantity', 'Prediction'], index=False)
    download_file = f'Files/prediction_{file}'
    return send_file(f'Files/prediction_{file}',
                     mimetype='text/csv',
                     attachment_filename='Predictions.csv',
                     as_attachment=True)


@app.route('/loginGoogle', methods=['POST'])
def loginWith():
    users = db.Users
    email = request.get_json()['email']
    name = request.get_json()['name']
    response = users.find_one({'name': name})
    if (not response):
        user_id = users.insert({
            'type': "Google Account Login",
            'email': email,
            'name': name
        })

    return jsonify({'result': "Successfully Added"})


@app.route('/sales', methods=['GET', 'POST'])
def predictSales():
    shutil.rmtree(app.config["UPLOAD_FOLDER"])
    os.mkdir(app.config["UPLOAD_FOLDER"])
    filesToUpload = request.files["file"]
    filesToUpload.save(os.path.join(app.config["UPLOAD_FOLDER"], filesToUpload.filename))
    result = testing(filesToUpload.filename)
    result = result.values.tolist()
    return jsonify({'result': result})


@app.route('/sales_one', methods=['GET', 'POST'])
def predictIndividual():
    day = request.get_json()['day']
    month = request.get_json()['month']
    year = request.get_json()['year']
    quantity = request.get_json()['quantity']
    result = testingIndividual(int(day), int(month), int(year), float(quantity))
    result = result.values.tolist()
    print(result)
    return jsonify({'result': result})


if __name__ == "__main__":
    app.run()
