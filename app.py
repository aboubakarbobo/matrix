import json
import ast
import re
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

def parse_matrix_input(input_str):
    """
    Parses matrix input from string into a 2D numpy array.
    """
    if not input_str or not input_str.strip():
        raise ValueError("L'entrée de la matrice ne peut pas être vide.")

    cleaned = input_str.strip()

    # Try literal eval / JSON first
    if cleaned.startswith('['):
        try:
            parsed = ast.literal_eval(cleaned)
            arr = np.array(parsed, dtype=float)
            if arr.ndim != 2:
                raise ValueError("La matrice doit être un tableau à 2 dimensions (ex: [[1, 2], [3, 4]]).")
            return arr
        except (ValueError, SyntaxError):
            pass

    # Parse line by line
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    matrix = []
    expected_cols = None

    for idx, line in enumerate(lines):
        # replace commas with spaces and split by whitespace
        row_str = line.replace(',', ' ')
        tokens = [t for t in row_str.split() if t]
        
        if not tokens:
            continue

        try:
            row = [float(val) for val in tokens]
        except ValueError:
            raise ValueError(f"Valeur non numérique détectée à la ligne {idx + 1}: '{line}'")

        if expected_cols is None:
            expected_cols = len(row)
        elif len(row) != expected_cols:
            raise ValueError(f"Incohérence des dimensions : la ligne {idx + 1} contient {len(row)} éléments alors que la première ligne en contient {expected_cols}.")

        matrix.append(row)

    if not matrix:
        raise ValueError("Impossible d'extraire des données de matrice valides.")

    return np.array(matrix, dtype=float)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/multiply', methods=['POST'])
def multiply():
    try:
        data = request.get_json(silent=True) or request.form

        raw_a = data.get('matrix_a', '')
        raw_b = data.get('matrix_b', '')

        matrix_a = parse_matrix_input(raw_a)
        matrix_b = parse_matrix_input(raw_b)

        rows_a, cols_a = matrix_a.shape
        rows_b, cols_b = matrix_b.shape

        if cols_a != rows_b:
            return jsonify({
                'success': False,
                'error': f"Multiplication impossible : la matrice A a {cols_a} colonne(s) "
                         f"alors que la matrice B a {rows_b} ligne(s). "
                         f"Le nombre de colonnes de A doit égaler le nombre de lignes de B (condition : A[m×n] × B[n×p])."
            }), 400

        result = np.dot(matrix_a, matrix_b)

        # Format numeric output nicely
        def clean_list(arr):
            out = []
            for row in arr.tolist():
                clean_row = [int(x) if x.is_integer() else round(x, 4) for x in row]
                out.append(clean_row)
            return out

        return jsonify({
            'success': True,
            'matrix_a': clean_list(matrix_a),
            'matrix_b': clean_list(matrix_b),
            'result': clean_list(result),
            'dimensions': {
                'a': f"{rows_a} × {cols_a}",
                'b': f"{rows_b} × {cols_b}",
                'result': f"{rows_a} × {cols_b}"
            }
        })

    except ValueError as ve:
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f"Erreur inattendue : {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
