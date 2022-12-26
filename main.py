import os
import random
import re
import traceback

import openai
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google.appengine.api import memcache, wrap_wsgi_app

from ask_embeddings import ask, ask_start

START_QUERY = "list some interesting key concepts, each on new line"
LIST_QUERY = "list some interesting key concepts related to {concept}, each on new line"
DESCRIBE_QUERY = "describe {concept}"
MAX_ITEMS_PER_LIST = 7
EMBEDDINGS_FILE = "embeddings/what-dimitri-learned.pkl"
WANDERING_MEMORY = 60 * 60 * 2  # 2 hours, why not
WANDERING_VARIETY = 5

app = Flask(__name__)
app.wsgi_app = wrap_wsgi_app(app.wsgi_app)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_TOKEN")


def get_memcache_entry(prefix, query, factory_function):
    variety = random.randint(0, WANDERING_VARIETY)
    key = f"{prefix}-{variety}-{query}"
    entry = memcache.get(key)
    if entry is None:
        entry = factory_function(query)
        memcache.set(key, entry, WANDERING_MEMORY)
    return entry


def sanitize(line):
    pattern = re.compile(r"^\d+\.")
    return pattern.sub("", line).lstrip("* ").lstrip("- ").strip()


def make_list(response):
    lines = response.splitlines()
    if len(lines) > MAX_ITEMS_PER_LIST:
        lines = random.sample(lines, MAX_ITEMS_PER_LIST)
    return [sanitize(line) for line in lines]


def ask_to_start(_):
    (response, issues) = ask_start(START_QUERY, EMBEDDINGS_FILE)
    return jsonify({
        "list": make_list(response),
        "issues": issues
    })


@app.route("/api/start", methods=["POST"])
def start():
    try:
        return get_memcache_entry("start", "", ask_to_start)

    except Exception as e:
        return jsonify({
            "error": f"{e}\n{traceback.print_exc()}"
        })


@app.route("/api/start", methods=["GET"])
def start_sample():
    return render_template("start.html")


def ask_to_describe(concept):
    (response, issues) = ask(DESCRIBE_QUERY.format(concept=concept), EMBEDDINGS_FILE)
    return jsonify({
        "text": response,
        "issues": issues
    })


@app.route("/api/describe", methods=["POST"])
def describe():
    concept = request.form["concept"]
    if not concept:
        return jsonify({
            "error": "Concept is required"
        })
    try:
        return get_memcache_entry("describe", concept, ask_to_describe)

    except Exception as e:
        return jsonify({
            "error": f"{e}\n{traceback.print_exc()}"
        })


@app.route("/api/describe", methods=["GET"])
def describe_sample():
    return render_template("describe.html")


def ask_to_list(concept):
    (response, issues) = ask(LIST_QUERY.format(concept=concept), EMBEDDINGS_FILE)
    return jsonify({
        "list": make_list(response),
        "issues": issues
    })


@app.route("/api/list", methods=["POST"])
def list():
    concept = request.form["concept"]
    if not concept:
        return jsonify({
            "error": "Concept is required"
        })
    try:
        return get_memcache_entry("list", concept, ask_to_list)

    except Exception as e:
        return jsonify({
            "error": f"{e}\n{traceback.print_exc()}"
        })


@app.route("/api/list", methods=["GET"])
def list_sample():
    return render_template("list.html")


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)
