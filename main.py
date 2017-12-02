import json
from flask import Flask, request, send_from_directory
app = Flask(__name__)


def summarize(query):
    return {'status': 'ok', 'result': query}


@app.route('/<path:path>')
def frontend(path):
    return send_from_directory('frontend', path)


@app.route('/')
def root():
    return frontend('index.html')


@app.route('/api/summarize', methods=['GET'])
def hello():
    query = request.args.get('query')

    if not query:
        return json.dumps({'status': 'error', 'reason': 'empty query'})

    return json.dumps(summarize(query))


if __name__ == '__main__':
    app.run(debug=True)