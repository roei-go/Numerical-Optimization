import numpy as np
import unittest
from src import utils
from src import unconstrained_min
import examples


class test_unconstrained_min(unittest.TestCase):

    def test_quad_min(self):
        x0 = np.array([[1], [1]])
        step_size = 0.1
        max_iter = 100
        param_tol = 1e-8
        obj_tol = 1e-12
        f = examples.quad_3()
        final_loc, success, path, obj_values = unconstrained_min.line_search(f, x0, step_size, obj_tol, param_tol, max_iter,dir_selection_method = 'nt',use_wolfe_step=True)
        # define the space over which to plot the objective contours
        # define the resolution (per axis)
        num_points = 100
        min_x1 = -1.1
        max_x1 = 1.1
        delta_x1 = (max_x1 - min_x1) / num_points
        x1 = np.arange(min_x1, max_x1, delta_x1)
        min_x2 = -1.1
        max_x2 = 1.1
        delta_x2 = (max_x2 - min_x2) / num_points
        x2 = np.arange(min_x2, max_x2, delta_x2)
        obj_path = path[2, :]
        levels = np.linspace(np.amin(obj_path),2*np.amax(obj_path))
        utils.plot_grad_desc(f, x1, x2, path[0:2,:],obj_path)
        utils.plot_obj_values(obj_values)
        self.assertTrue(success)

    def test_rosenbrock_min(self):
        x0 = np.array([[2], [2]])
        step_size = 0.001
        max_iter = 10000
        param_tol = 1e-8
        obj_tol = 1e-7
        final_loc, success, path, obj_values = unconstrained_min.line_search(examples.rosenbrock, x0, step_size, obj_tol, param_tol, max_iter,dir_selection_method = 'bfgs',use_wolfe_step=True)
        # define the space over which to plot the objective contours
        # define the resolution (per axis)
        num_points = 100
        # get the ranges of coordinates spanned by the path
        half_int = 0.1
        min_x1 = -2.5
        max_x1 = 2.5
        delta_x1 = (max_x1 - min_x1) / num_points
        x1 = np.arange(min_x1, max_x1, delta_x1)
        min_x2 = -5
        max_x2 = 5
        delta_x2 = (max_x2 - min_x2) / num_points
        x2 = np.arange(min_x2, max_x2, delta_x2)
        obj_path = path[2, :]
        levels = np.linspace(np.amin(obj_path), np.amax(obj_path))
        utils.plot_grad_desc(examples.rosenbrock, x1, x2, path[0:2, :], levels)
        utils.plot_obj_values(obj_values)
        self.assertTrue(success)



    def test_lin_min(self):
        x0 = np.array([[2], [2]])
        step_size = 0.1
        max_iter = 100
        param_tol = 1e-8
        obj_tol = 1e-12
        f = examples.lin_testcase()
        final_loc, success, path, obj_values = unconstrained_min.line_search(f, x0, step_size, obj_tol,
                                                                      param_tol, max_iter)
        # define the space over which to plot the objective contours
        # define the resolution (per axis)
        num_points = 100
        min_x1 = -10
        max_x1 = 3
        delta_x1 = (max_x1 - min_x1) / num_points
        x1 = np.arange(min_x1, max_x1, delta_x1)
        min_x2 = -10
        max_x2 = 3
        delta_x2 = (max_x2 - min_x2) / num_points
        x2 = np.arange(min_x2, max_x2, delta_x2)
        obj_path = path[2, :]
        levels = np.linspace(np.amin(obj_path), np.amax(obj_path))
        utils.plot_grad_desc(f, x1, x2, path[0:2, :], levels)
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
