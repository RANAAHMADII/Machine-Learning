import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from cvxopt import matrix, solvers
parameters = {}
KERNEL_LINEAR = 1
KERNEL_RBF = 2

DATASET_LINEARLY_SEPARABLE = 1
DATASET_CIRCULAR = 2

def generate_data(dataset):
    n = 2000
    X = np.random.rand(n, 2)
    y = np.zeros((n,))
    noise = np.random.uniform(-0.05, 0.05, n)
    
    if dataset == DATASET_LINEARLY_SEPARABLE:
        for i in range(n):
            x1 = X[i][0]
            x2 = X[i][1] + noise[i]
            y[i] = 1.0 if x2 <= 1.0 * x1 else -1.0
    else:
        r = 0.3
        centre = np.array([0.5, 0.5])

        for i in range(n):
            dist = np.linalg.norm(X[i] - centre) + noise[i]
            y[i] = 1.0 if dist <= r else -1.0

    parameters['X'] = X
    parameters['y'] = y
    
    return X, y

X, y = generate_data(DATASET_LINEARLY_SEPARABLE)
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(X[:,0], X[:,1], edgecolors=['red' if y_i == -1 else 'blue' for y_i in y], facecolors='none', s=30)
plt.show()


def gram_matrix(X, Y, kernel_type, gamma=0.5):
    K = np.zeros((X.shape[0], Y.shape[0]))
    if kernel_type == KERNEL_LINEAR:
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = np.dot(x.T, y)
                
    elif kernel_type == KERNEL_RBF:
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = np.exp(-gamma * np.linalg.norm(x - y) ** 2)          
    return K

def train_svm(kernel):
    C = 100
    n, k = X.shape
    
    y_matrix = y.reshape(1, -1)
    
    H = np.dot(y_matrix.T, y_matrix) * gram_matrix(X, X, kernel)
    P = matrix(H)
    q = matrix(-np.ones((n, 1)))
    G = matrix(np.vstack((-np.eye((n)), np.eye(n))))
    h = matrix(np.vstack((np.zeros((n,1)), np.ones((n,1)) * C)))
    A = matrix(y_matrix)
    b = matrix(np.zeros(1))
    
    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    solvers.options['feastol'] = 1e-10

    return solvers.qp(P, q, G, h, A, b)

X = parameters['X']
svm_parameters = train_svm(KERNEL_LINEAR)
print(svm_parameters)

def get_parameters(alphas):
    threshold = 1e-5 # Values greater than zero (some floating point tolerance)
    S = (alphas > threshold).reshape(-1, )
    w = np.dot(X.T, alphas * y)
    b = y[S] - np.dot(X[S], w) # b calculation
    b = np.mean(b)
    return w, b, S

alphas = np.array(svm_parameters['x'])[:, 0]
w, b, S = get_parameters(alphas)

print('Alphas:', alphas[S][0:20])
print('w and b', w, b)

def sv_graph():
    support_vectors = X[S]
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(X[:,0], X[:,1], edgecolors=['red' if y_i == -1 else 'blue' for y_i in y], facecolors='none', s=30)
    ax.scatter(support_vectors[:,0], support_vectors[:,1], c='black', s=50)
    plt.show()
    
sv_graph()

X, y = generate_data(DATASET_CIRCULAR)

svm_parameters = train_svm(KERNEL_RBF)

alphas = np.array(svm_parameters['x'])[:, 0]
w, b, S = get_parameters(alphas)

print('Alphas:', alphas[S][0:20])
print('w and b', w, b)
sv_graph()
plt.savefig('2.png',dpi = 300)
