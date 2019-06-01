import json
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from classifier import GeneralClassifier
from classifier import ConfidentialClassifier

app = Flask(__name__)
CORS(app)


@app.route("/gen_category", methods=['POST'])
def return_gen_category():

    if (request.method == 'POST'):
        inStream = request.get_json(force=True)
        results = []
        for obj in inStream:
            path = obj['path']
            text = obj['text']
            category = GeneralClassifier().classify(text)
            category_dict = {
                'path': path,
                'category': category
            }
            results.append(category_dict)
            print(results)
            print(jsonify(results))
        return jsonify(results)


@app.route("/conf_category", methods=['POST'])
def return_conf_category():
    if (request.method == 'POST'):
        inStream = request.get_json(force=True)
        results = []
        for obj in inStream:
            path = obj['path']
            text = obj['text']
            category = ConfidentialClassifier().classify(text)
            category_dict = {
                'path': path,
                'category': category
            }
            results.append(category_dict)
        return jsonify(results)


@app.route("/", methods=['GET', 'POST'])
def default():
    return "<h1> Welcome to Classifer - The Document Classifier <h1>"


if __name__ == "__main__":
    app.run()
