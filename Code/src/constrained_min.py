import numpy as np
from src import utils


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, newton_dec_tol=1e-6, barrier_tol=1e-8, inner_loop_max_iter=100):
    # init t,mu
    t = 1
    mu = 10
    # the number of inequality constraints
    m = len(ineq_constraints)
    # the dimension of the solution space
    n = x0.shape[0]
    # the number of equality constraints
    if eq_constraints_mat is not None:
        p = eq_constraints_mat.shape[0]
    else:
        p = 0
        eq_constraints_mat = np.empty((p,n))
    print("solving optimization problem with %d inequality constraints and %d equality constraints" % (m,p))
    x_k = x0.copy()
    # init the path
    unconstrained_obj_k, _, _ = func(x_k)
    path = np.concatenate((x_k,unconstrained_obj_k), axis=0)
    # outer loop
    while m/t > barrier_tol:

        # in the inner loop, we use Newton's method for minimizing t*f_0(x) + phi(x)
        i = 0
        print("outer iteration %d, t is %f" %(i,t))
        while i < inner_loop_max_iter:
            i += 1
            obj, obj_grad, obj_hess = inner_loop_obj(x_k, func, ineq_constraints, t)
            # build the system for Newton's method with equality constraints
            B = np.block([[obj_hess          , eq_constraints_mat.T],
                          [eq_constraints_mat, np.zeros((p,p))    ]])
            # build the right hand side - note that in this solver we assume a feasible starting point, so the bottom block of the RHS
            # is just 0 (instead of the residual), i.e. the location x_k in this iteration should satisfy A x_k = 0
            rhs = np.block([[-obj_grad      ],
                            [np.zeros((p,1))]])
            # solve the system to obtain the newton direction
            solve = np.linalg.solve(B, rhs)
            # obtain the newton direction - this is the upper part of the solution
            ndir = solve[:n,:]
            # calculate the Newton decrement lambda
            newton_lambda = np.sqrt(np.dot(ndir.T, np.dot(obj_hess,ndir)))
            # check convergence of the inner loop
            if newton_lambda < newton_dec_tol:
                break
            # choose the step size using the first Wolfe condition + backtracking
            alpha = 1
            back_track_factor = 0.9
            slope_ratio = 1e-4
            # first verify feasibility with regard to the inequality constraints
            x_kp1 = x_k + alpha * ndir
            while not is_feasible(x_kp1,ineq_constraints):
                alpha = alpha * back_track_factor
                x_kp1 = x_k + alpha * ndir
            # now continue to backtrack on alpha
            f_x_kp1, _, _ = inner_loop_obj(x_kp1, func, ineq_constraints, t)
            while f_x_kp1 > obj + slope_ratio * alpha * np.dot(obj_grad.T, ndir):
                alpha = alpha * back_track_factor
                x_kp1 = x_k + alpha * ndir
                f_x_kp1, _, _ = inner_loop_obj(x_kp1, func, ineq_constraints, t)
            print("interior_pt inner loop:")
            # find the unconstrained objective value
            unconstrained_obj_kp1, _, _ = func(x_kp1)
            utils.report_iter(i, x_k, x_kp1, unconstrained_obj_k, unconstrained_obj_kp1)
            # update the path
            path = np.concatenate((path, np.concatenate((x_kp1, unconstrained_obj_kp1), axis=0)), axis=1)
            # update the location for the next iteration
            x_k = x_kp1.copy()
            unconstrained_obj_k = unconstrained_obj_kp1
        # update t for the next iteration of the outer loop
        t *= mu

    # find the objective value at the last location
    f_x, f_grad, _ = func(x_k)
    return x_k, path
    


# define a helper function that will return the gradient and hessian of the logarithmic barrier: phi(x)
def phi(x, ineq_constraints):

    phi_x = 0
    phi_grad = np.zeros((x.shape[0],1))
    phi_hess = np.zeros((x.shape[0],x.shape[0]))
    # going over all inequality functions
    for i in range(len(ineq_constraints)):
        # each function should return the value, the gradient and the Hessian for a given point x
        f, grad, H = ineq_constraints[i](x,generate_hessian=True)

        phi_x += -np.log(-f)
        phi_grad += grad/(-f)
        phi_hess += (1/(f**2))* np.dot(grad,grad.T) + (1/-f) * H
    return  phi_x, phi_grad, phi_hess

# define a function that returns the function to be minimized in the inner loop - i.e. the unconstrained objective plus the log barrier
def inner_loop_obj(x, func, ineq_constraints, t):
    phi_x, phi_grad, phi_hess = phi(x, ineq_constraints)
    f_x, f_grad, f_hess = func(x,generate_hessian=True)
    obj = t * f_x + phi_x
    obj_grad = t * f_grad + phi_grad
    obj_hess = t * f_hess + phi_hess

    return obj, obj_grad, obj_hess

def is_feasible(x,ineq_constraints):
    """Check if a point satisfies all inequality constraints"""
    feasible = True
    for i in range(len(ineq_constraints)):
        f, _, _ = ineq_constraints[i](x)
        feasible = feasible and (f.squeeze() <= 0)
    return feasible