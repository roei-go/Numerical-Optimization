import numpy as np
from src import utils


def line_search(f, x0, step_size, obj_tol, param_tol, max_iter,dir_selection_method = 'gd',init_step_len=1,slope_ratio=1e-4,back_track_factor=0.2,use_wolfe_step=False):

    # initialize iteration counter
    i = 0
    x_k = x0
    f_x_k, df_x_k,H_x_k = f(x_k,generate_hessian=(dir_selection_method=='nt' or dir_selection_method=='bfgs'))
    # for BFGS, the initial approximation to the Hessian mat may be the Hessian itself. Here we choose the identity
    # matrix
    if dir_selection_method=='bfgs':
        #B_k = H_x_k.copy()
        B_k = np.eye(x0.shape[0])

    # init the path
    path = np.concatenate((x_k,f_x_k), axis=0)
    # init a list of objective values
    obj_values = [f_x_k]

    success = False

    while (not success) and (i < max_iter):

        if dir_selection_method=='gd':
            direction = -df_x_k

        elif dir_selection_method=='nt':
            direction = newton_dir(df_x_k, H_x_k)

        elif dir_selection_method=='bfgs':
            direction = bfgs_dir(df_x_k,B_k)
        else:
            assert "Invalid direction selection method given"
        # calculate step size
        if use_wolfe_step:
            step_size = find_step_size(f, x_k, direction, init_step_len, slope_ratio, back_track_factor)
        # update the location
        x_kp1 = x_k + step_size * direction
        # update objective, gradient and Hessian for the next iteration
        f_x_kp1, df_x_kp1, H_x_kp1 = f(x_kp1,dir_selection_method=='nt')
        # update the path
        path = np.concatenate((path, np.concatenate((x_kp1,f_x_kp1), axis=0)), axis=1)
        i += 1
        utils.report_iter(i, x_k, x_kp1, f_x_k, f_x_kp1)
        success = utils.check_conv(x_kp1, x_k, f_x_kp1, f_x_k, df_x_kp1, obj_tol, param_tol)
        obj_values.append(f_x_kp1.squeeze())
        if dir_selection_method == 'bfgs':
            B_k = update_bfgs(x_k, x_kp1, df_x_k, df_x_kp1, B_k).copy()
        # rename the current location, grad and Hessian to be the starting point in the next iteration
        x_k = x_kp1.copy()
        f_x_k = f_x_kp1.copy()
        df_x_k = df_x_kp1.copy()
        H_x_k = H_x_kp1
    if success:
        print("line search converged successfully")
    else:
        print("line search failed to converge ")
    return x_k, success, path, obj_values


def newton_dir(df_x_k,H_x_k):
    new_dir = np.linalg.solve(H_x_k,-df_x_k)
    # check that it's a descent direction
    assert np.dot(new_dir.T,df_x_k) < 0, "In newton_dir: new direction is not a descent direction"
    return new_dir


def bfgs_dir(df_x_k, B_k):
    new_dir = np.linalg.solve(B_k, -df_x_k)
    return new_dir

def update_bfgs(x_k,x_kp1,df_x_k,df_x_kp1,B_k):
    # update the Hessian approximation for the next iteration
    s_k = x_kp1 - x_k
    y_k = df_x_kp1 - df_x_k
    B_kp1 = B_k - np.dot(np.dot(B_k, s_k), np.dot(s_k.T, B_k)) / (np.dot(np.dot(s_k.T, B_k), s_k)) + np.dot(y_k, y_k.T) / np.dot(y_k.T, s_k)
    return B_kp1

def find_step_size(f,x_k,direction,init_step_len,slope_ratio,back_track_factor):
    alpha = init_step_len
    f_x_k, df_x_k, _ = f(x_k)
    x_kp1 = x_k + alpha * direction
    f_x_kp1, _, _ = f(x_kp1)
    while f_x_kp1 > f_x_k + slope_ratio*alpha*np.dot(df_x_k.T,direction):
        alpha = alpha*back_track_factor
        x_kp1 = x_k + alpha*direction
        f_x_kp1, _, _ = f(x_kp1)

    return alpha


