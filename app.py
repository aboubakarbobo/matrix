import json
import ast
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

def parse_matrix_input(input_str):
    if not input_str or not input_str.strip(): raise ValueError("Entrée vide.")
    cleaned = input_str.strip()
    # Gestion des listes de listes (format A)
    if isinstance(input_str, list): return np.array(input_str, dtype=float)
    if cleaned.startswith('['):
        try:
            return np.array(ast.literal_eval(cleaned), dtype=float)
        except: pass
    # Gestion du format texte matriciel
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    matrix = []
    for line in lines:
        matrix.append([float(x) for x in line.replace(',', ' ').split()])
    return np.array(matrix, dtype=float)

def clean_list(arr):
    if np.isscalar(arr): return int(arr) if float(arr).is_integer() else round(float(arr), 4)
    return [[int(x) if float(x).is_integer() else round(float(x), 4) for x in row] for row in arr.tolist()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        op = data.get('operation', 'mul')
        # La réception est différente si l'input est structuré comme dans le HTML actuel
        A = parse_matrix_input(data.get('matrix_a'))
        B = parse_matrix_input(data.get('matrix_b')) if data.get('matrix_b') else None
        
        if op == 'add': res = A + B
        elif op == 'sub': res = A - B
        elif op == 'mul': res = np.dot(A, B)
        elif op == 'transpose': res = A.T
        elif op == 'det': res = np.linalg.det(A)
        elif op == 'inv': res = np.linalg.inv(A)
        elif op == 'pow': res = np.linalg.matrix_power(A, int(data.get('power', 1)))
        elif op == 'scalar': res = float(data.get('k', 1)) * A
        else: raise ValueError("Opération inconnue")

        return jsonify({'success': True, 'result': clean_list(res), 'is_scalar': np.isscalar(res) or isinstance(res, (float, int))})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
