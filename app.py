from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pyAgrum as gum
import json

app = Flask(__name__)
CORS(app)

print('Server started. Loading model from bif file...')

bn = gum.loadBN('assets/bayes_credit_k2_hill_climb_with_restricted.bif')
    
print('model loaded from bif file')

inference = gum.VariableElimination(bn)

@app.route("/", methods=['GET'])
def index():
    routes = {}
    for route in app.url_map.iter_rules():
        routes[route.endpoint] = f"{request.url_root.strip('/')}{route}"
    return make_response(jsonify(routes), 200)

@app.route("/network", methods=['GET'])
def network():
    nodes = bn.names()
    states = {node: bn.variable(node).labels() for node in nodes}
    return make_response(jsonify(states), 200)

@app.route("/edges", methods=['GET'])
def edges():
    arcs = bn.arcs()
    edges = [(bn.variable(arc[0]).name(), bn.variable(arc[1]).name()) for arc in arcs]
    return make_response(jsonify(edges), 200)

@app.route("/inference", methods=['GET'])
def inf():
    req_variables = request.args.getlist('query')
    req_evidence = request.args.get('evidence')

    if req_evidence == None or len(req_evidence) == 0:
        req_evidence = {}
    else:
        req_evidence = json.loads(req_evidence)
    
    result = {}
    for variable in req_variables:
        inference.setEvidence(req_evidence)
        inference.makeInference()
        query = inference.posterior(variable)
        result[variable] = dict(zip(bn.variable(variable).labels(), query.tolist()))
    
    return make_response(jsonify(result), 200)

@app.errorhandler(Exception)
def handle_exception(e):
    return make_response(jsonify(error=str(e)), 500)

if __name__ == "__main__":
    app.json.sort_keys = False
    app.run()