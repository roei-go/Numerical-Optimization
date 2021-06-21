import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def check_conv(x_next, x_prev, f_next, f_prev, df_next, obj_tol, param_tol):
    delta_loc = np.linalg.norm(x_next - x_prev)
    delta_obj = np.abs(f_next - f_prev)
    # declaring convergence in the below cases
    # 1. The next location is close enough to the current location
    # 2. The next objective value is close enough to the current one
    # 3. The gradient is 0
    if delta_loc <= param_tol:
        print("successful convergence: next location is close to current one")
        return True

    elif delta_obj <= obj_tol:
        print("successful convergence: next objective value is close to current one")
        return True

    elif np.linalg.norm(df_next) == 0:
        print("successful convergence: gradient is 0")
        return True

    else:
        return False


def report_iter(iter_cnt, prev_loc, curr_loc, prev_obj, curr_obj):
    print("in iteration %d" % iter_cnt)
    print("----------------------------------------------------------------------------------------------------")
    print("current location is %s" % str(curr_loc.T))
    print("current objective value is %.4E" % curr_obj)
    print("step length taken is %.4E" % np.linalg.norm(curr_loc - prev_loc))
    print("objective change is %.4E \n\n\n" % (np.abs(curr_obj - prev_obj)))

    return None


def plot_grad_desc(f, x1, x2, path, path_levels):

    X1, X2 = np.meshgrid(x1, x2)
    min_x1 = np.amin(x1)
    max_x1 = np.amax(x1)
    min_x2 = np.amin(x2)
    max_x2 = np.amax(x2)
    c1 = np.array([[min_x1], [min_x2]])
    c2 = np.array([[min_x1], [max_x2]])
    c3 = np.array([[max_x1], [max_x2]])
    c4 = np.array([[max_x1], [min_x2]])
    # get the function value in the 4 corners of the grid
    f1, _, _ = f(c1)
    f2, _, _ = f(c2)
    f3, _, _ = f(c3)
    f4, _, _ = f(c4)
    min_path_levels = np.amin(path_levels)
    max_path_levels = np.amax(path_levels)

    min_f = np.amin(np.array([np.squeeze(f1), np.squeeze(f2), np.squeeze(f3), np.squeeze(f4), np.squeeze(min_path_levels)]))
    max_f = np.amax(np.array([np.squeeze(f1), np.squeeze(f2), np.squeeze(f3), np.squeeze(f4), np.squeeze(max_path_levels)]))
    int_levels = np.linspace(min_f, max_f,num=30)
    Z = np.zeros(X1.shape)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([[X1[i,j]],[X2[i,j]]])
            Z[i, j] = f(x)[0]
    fig, ax = plt.subplots()
    CS = ax.contour(X1, X2, Z, levels=int_levels)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    # make a colorbar for the contour lines
    CB = fig.colorbar(CS)
    ax.plot(path[0,:-1], path[1,:-1], 'bo')
    ax.plot(path[0, -1], path[1, -1], 'r*')
    plt.show()


def plot_obj_values(obj_values):
    fig, ax = plt.subplots(figsize=(6,6))
    iterations = list(range(len(obj_values)))
    ax.plot(iterations,obj_values)
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective")
    ax.set_title("Objective values along iterations")
    plt.show()

