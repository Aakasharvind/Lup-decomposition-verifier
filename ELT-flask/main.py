from flask import Flask, render_template, request
import numpy as np
from scipy.linalg import lu_factor, lu_solve

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # Get the matrix A and vector b from the form data
    A = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            A[i][j] = float(request.form['a'+str(i+1)+str(j+1)])
            
    b = np.array([float(request.form['b1']), float(request.form['b2']), float(request.form['b3'])])
    
    # Perform LUP decomposition on the coefficient matrix A
    lu, piv = lu_factor(A)
    L = np.tril(lu, k=-1) + np.eye(len(A))
    U = np.triu(lu)
    P = np.eye(len(A))[:, piv]

    # Solve the linear system of equations Ax = b using LUP decomposition
    x = lu_solve((lu, piv), b)

    # Render the results template with the solution
    return render_template('result.html', L=L, U=U, P=P, x=x)

if __name__ == '__main__':
    app.run(debug=True)
