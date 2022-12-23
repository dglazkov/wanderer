import os
import re
import traceback
import random


from ask_embeddings import ask
import openai
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

START_QUERY = "list some interesting key concepts, each on new line"
LIST_QUERY = "list some interesting key concepts related to {concept}, each on new line"
DESCRIBE_QUERY = "describe {concept}"
MAX_ITEMS_PER_LIST = 7

app = Flask(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_TOKEN")

def sanitize(line):
    pattern = re.compile(r"^\d+\.")
    return pattern.sub("", line).lstrip("* ").lstrip("- ").strip()


def make_list(response):
    lines = random.sample(response.splitlines(), MAX_ITEMS_PER_LIST)
    return [sanitize(line) for line in lines]
    
@app.route("/api/start", methods=["POST"])
def start():
    try:
        response = ask(START_QUERY, "embeddings/what-dimitri-learned.pkl")
        return jsonify({
            "list": make_list(response)
        })

    except Exception as e:
        return jsonify({
            "error": f"{e}\n{traceback.print_exc()}"
        })


@app.route("/api/start", methods=["GET"])
def start_sample():
    return render_template("start.html")


@app.route("/api/describe", methods=["POST"])
def describe():
    concept = request.form["concept"]
    if not concept:
        return jsonify({
            "error": "Concept is required"
        })
    try:
        response = ask(DESCRIBE_QUERY.format(concept=concept),
                       "embeddings/what-dimitri-learned.pkl")
        return jsonify({
            "text": response
        })

    except Exception as e:
        return jsonify({
            "error": f"{e}\n{traceback.print_exc()}"
        })


@app.route("/api/describe", methods=["GET"])
def describe_sample():
    return render_template("describe.html")


@app.route("/api/list", methods=["POST"])
def list():
    concept = request.form["concept"]
    if not concept:
        return jsonify({
            "error": "Concept is required"
        })
    try:
        response = ask(LIST_QUERY.format(concept=concept),
                       "embeddings/what-dimitri-learned.pkl")
        return jsonify({
            "list": make_list(response)
        })

    except Exception as e:
        return jsonify({
            "error": f"{e}\n{traceback.print_exc()}"
        })


@app.route("/api/list", methods=["GET"])
def list_sample():
    return render_template("list.html")


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)
