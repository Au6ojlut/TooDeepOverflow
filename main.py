import json
from flask import Flask, request, send_from_directory, make_response
from deepoverflow import Application
from functools import wraps, update_wrapper
import datetime
app = Flask(__name__)

do = Application('/home/egor/ml/DeepOverflow/data')


def summarize(query):
    return {'status': 'ok', 'result': do.process(query)}


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response


    return update_wrapper(no_cache, view)


@app.route('/<path:path>')
@nocache
def frontend(path):
    return send_from_directory('frontend', path)


@app.route('/')
@nocache
def root():
    return frontend('index.html')


@app.route('/api/summarize', methods=['post'])
@nocache
def api():
    if 'query' not in request.form:
        return json.dumps({'status': 'error', 'reason': 'query not specified'})

    query = request.form['query']

    if not query:
        return json.dumps({'status': 'error', 'reason': 'empty query'})

    return json.dumps(summarize(query))


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')