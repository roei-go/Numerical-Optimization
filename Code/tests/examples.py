import numpy as np


class quad:
    """
    implements a single quadratic function of the form f = x.T Q x + b.T x + c
    """
    def __init__(self,Q,b,c):
        self.Q = Q
        self.b = b
        self.c = c

    def func(self, x, generate_hessian=False):
        if generate_hessian:
            return x.T @ self.Q @ x + self.b.T @ x + self.c, 2 * self.Q @ x + self.b, 2 * self.Q
        else:
            return x.T @ self.Q @ x + self.b.T @ x + self.c, 2 * self.Q @ x + self.b, None


def quad_1():
    Q = np.array([[1, 0], [0, 1]])
    b = np.zeros((Q.shape[0],1))
    c = 0
    quad_inst = quad(Q, b, c)
    return quad_inst.func


def quad_2():
    Q = np.array([[5, 0], [0, 1]])
    b = np.zeros((Q.shape[0], 1))
    c = 0
    quad_inst = quad(Q, b, c)
    return quad_inst.func


def quad_3():
    A = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    B = np.array([[5, 0], [0, 1]])
    Q = np.dot(np.dot(A.T, B), A)
    b = np.zeros((Q.shape[0], 1))
    c = 0
    quad_inst = quad(Q, b, c)
    return quad_inst.func


def rosenbrock(x, generate_hessian=False):
    x1 = x[0,0]
    x2 = x[1,0]
    f = np.array([[100*np.power(x2 - x1**2,2) + np.power(1-x1, 2)]])
    grad = np.array([[400*np.power(x1, 3) + 2*x1 - 400*x1*x2 - 2], [200*(x2 - np.power(x1, 2))]])
    if generate_hessian:
        H = np.array([[1200*np.power(x1, 2) - 400*x2 + 2 , -400*x1], [-400*x1 , 200]])
        return f, grad, H
    else:
        return f, grad, None


def lin_testcase():
    a = np.ones((2,1))
    f = lin(a=a, b=0)
    return f.func


def constrained_lin_testcase():
    # define the objective function of the form <a.T,x>. note that a is defined here as [-1,-1] because the corresponding
    # problem is maximizing x + y
    a = -np.ones((2,1))
    f_0 = lin(a=a, b=0)
    # define the constraints
    f_1 = lin(a=a, b=1)                    # -x -y +1 < 0
    f_2 = lin(a=np.array([[0], [1]]), b=-1) # y - 1 < 0
    f_3 = lin(a=np.array([[1], [0]]), b=-2) # x - 2 < 0
    f_4 = lin(a=np.array([[0], [-1]]), b=0) # -y < 0
    ineq_constraints = [f_1.func, f_2.func, f_3.func, f_4.func]
    eq_constraints_mat = None
    eq_constraints_rhs = None

    return f_0.func,  ineq_constraints, eq_constraints_mat, eq_constraints_rhs

def constrained_quad_testcase():
    Q = np.eye(3)
    b = np.array([[0], [0], [2]])
    c = 1
    f_0 = quad(Q, b, c)
    # define the constraints
    f_1 = lin(a=np.array([[-1], [0], [0]]), b=0)  # -x < 0
    f_2 = lin(a=np.array([[0], [-1], [0]]), b=0)  # -y < 0
    f_3 = lin(a=np.array([[0], [0], [-1]]), b=0)  # -z < 0

    ineq_constraints = [f_1.func, f_2.func, f_3.func]
    eq_constraints_mat = np.ones((1,3))
    eq_constraints_rhs = np.ones((1,1))

    return f_0.func,  ineq_constraints, eq_constraints_mat, eq_constraints_rhs

class lin:
    """
    implements a single linear function of the form f = <a.T,x> + b
    """
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def func(self, x, generate_hessian=False):
        if generate_hessian:
            H = np.zeros((self.a.shape[0], self.a.shape[0]))
            return np.dot(self.a.T, x) + self.b, self.a, H
        else:
            return np.dot(self.a.T, x) + self.b, self.a,None
