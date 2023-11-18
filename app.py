from flask import Flask, request, jsonify
from flask_cors import CORS
import pyAgrum as gum
import json

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    nodes = bn.names()
    states = {node: bn.variable(node).labels() for node in nodes}
    return jsonify(states)

@app.route("/network")
def network():
    nodes = bn.names()
    states = {node: bn.variable(node).labels() for node in nodes}
    return jsonify(states)

@app.route("/edges")
def edges():
    arcs = bn.arcs()
    edges = [(bn.variable(arc[0]).name(), bn.variable(arc[1]).name()) for arc in arcs]
    return jsonify(edges)

@app.route("/inference")
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
        
    return result

if __name__ == "__main__":
    print('Server started. Loading model from bif file...')

    bn = gum.loadBN('assets/bayes_credit_k2_hill_climb_with_restricted.bif')
    
    print('model loaded from bif file')

    inference = gum.VariableElimination(bn)
    app.run()