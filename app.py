import json
import ast
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

def parse_matrix_input(input_str):
    if not input_str or not input_str.strip(): raise ValueError("Entrée vide.")
    cleaned = input_str.strip()
    if cleaned.startswith('['):
        try:
            parsed = ast.literal_eval(cleaned)
            arr = np.array(parsed, dtype=float)
            if arr.ndim != 2: raise ValueError("Doit être 2D.")
            return arr
        except (ValueError, SyntaxError): pass
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    matrix = []
    expected_cols = None
    for idx, line in enumerate(lines):
        row_str = line.replace(',', ' ')
        tokens = [t for t in row_str.split() if t]
        if not tokens: continue
        try:
            row = [float(val) for val in tokens]
        except ValueError: raise ValueError(f"Valeur non numérique ligne {idx + 1}")
        if expected_cols is None: expected_cols = len(row)
        elif len(row) != expected_cols: raise ValueError("Incohérence des dimensions")
        matrix.append(row)
    if not matrix: raise ValueError("Données invalides.")
    return np.array(matrix, dtype=float)

def clean_list(arr):
    if np.isscalar(arr): return int(arr) if float(arr).is_integer() else round(float(arr), 4)
    out = []
    for row in arr.tolist():
        clean_row = [int(x) if float(x).is_integer() else round(float(x), 4) for x in row]
        out.append(clean_row)
    return out

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json(silent=True) or request.form
        op = data.get('operation')
        raw_a = data.get('matrix_a', '')
        A = parse_matrix_input(raw_a)
        B = None
        if op in ['add', 'sub', 'mul']:
            B = parse_matrix_input(data.get('matrix_b', ''))

        res = None
        if op == 'add':
            if A.shape != B.shape: raise ValueError("Dimensions incompatibles.")
            res = A + B
        elif op == 'sub':
            if A.shape != B.shape: raise ValueError("Dimensions incompatibles.")
            res = A - B
        elif op == 'mul':
            if A.shape[1] != B.shape[0]: raise ValueError("Colonnes de A != Lignes de B.")
            res = np.dot(A, B)
        elif op == 'transpose': res = A.T
        elif op == 'det':
            if A.shape[0] != A.shape[1]: raise ValueError("La matrice doit être carrée.")
            res = np.linalg.det(A)
        elif op == 'inv':
            if A.shape[0] != A.shape[1]: raise ValueError("La matrice doit être carrée.")
            res = np.linalg.inv(A)
        elif op == 'pow':
            if A.shape[0] != A.shape[1]: raise ValueError("La matrice doit être carrée.")
            res = np.linalg.matrix_power(A, int(data.get('power', 1)))
        elif op == 'scalar':
            res = float(data.get('k', 1)) * A

        return jsonify({'success': True, 'result': clean_list(res), 'is_scalar': np.isscalar(res) or isinstance(res, (float, int))})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
