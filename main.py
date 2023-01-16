import json
import os
import random
import re
import traceback

import openai
import urllib3
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google.appengine.api import memcache, wrap_wsgi_app

import polymath

START_QUERY = "list some interesting key concepts, each on new line"
LIST_QUERY = "list some interesting key concepts related to {concept}, each on new line"
DESCRIBE_QUERY = "describe {concept}"
MAX_ITEMS_PER_LIST = 7
POLYMATH_SERVER = "polymath.glazkov.com"
WANDERING_MEMORY = 60 * 60 * 2  # 2 hours, why not
WANDERING_VARIETY = 5
CONTEXT_TOKEN_COUNT = 1500

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


def query_polymath_server(query_embedding, random, server):
    http = urllib3.PoolManager()
    fields = {
        "version": polymath.CURRENT_VERSION,
        "query_embedding_model": polymath.EMBEDDINGS_MODEL_ID,
        "count": CONTEXT_TOKEN_COUNT
    }
    if random:
        fields["sort"] = "random"
    else:
        fields["query_embedding"] = query_embedding
    response = http.request(
        'POST', server, fields=fields).data
    obj = json.loads(response)
    if 'error' in obj:
        error = obj['error']
        raise Exception(f"Server returned an error: {error}")
    return polymath.Library(data=obj)


def ask(query, random=False):
    query_vector = None if random else polymath.base64_from_vector(
        polymath.get_embedding(query))

    library = query_polymath_server(query_vector, random, POLYMATH_SERVER)
    context = polymath.get_context_for_library(library)
    sources = polymath.get_chunk_infos_for_library(library)
    completion = polymath.get_completion_with_context(query, context)

    return (completion, sources[:3])


def ask_to_start(_):
    (response, issues) = ask(START_QUERY, True)
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
    (response, issues) = ask(DESCRIBE_QUERY.format(concept=concept))
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
    (response, issues) = ask(LIST_QUERY.format(concept=concept))
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
