from flask import Flask, request, jsonify
from flask_cors import CORS
from pgmpy.inference import VariableElimination
from pgmpy.readwrite.BIF import BIFReader
import json

app = Flask(__name__)
CORS(app)

@app.route("/network")
def network():
    return jsonify(bn.states)

@app.route("/edges")
def edges():
    return list(bn.edges())

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
        query = inference.query(variables=[variable], evidence=req_evidence)
        result[variable] = dict(zip(query.state_names[variable], query.values))
        
    return result

if __name__ == "__main__":
    print('Server started. Loading model from bif file...')

    reader = BIFReader('assets/bayes_credit_k2_hill_climb_with_restricted.bif')
    
    print('model loaded from bif file')

    bn = reader.get_model()
    print('model deserialized')

    inference = VariableElimination(bn)
    app.run()