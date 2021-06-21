import unittest
import numpy as np
from src import constrained_min
import examples
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # appropriate import to draw 3d polygons
from matplotlib import style



class test_constrained_min(unittest.TestCase):
    def test_lp(self):
        x0 = np.array([[0.5], [0.75]])
        f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs = examples.constrained_lin_testcase()
        x_s, path = constrained_min.interior_pt(f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)


        # plotting
        f_plot = examples.lin(a=np.ones((2,1)),b=0).func
        num_points = 400
        min_x1 = -1.1
        max_x1 = 2.1
        delta_x1 = (max_x1 - min_x1) / num_points
        x1 = np.arange(min_x1, max_x1, delta_x1)
        min_x2 = -0.1
        max_x2 = 1.1
        delta_x2 = (max_x2 - min_x2) / num_points
        x2 = np.arange(min_x2, max_x2, delta_x2)


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
        f1, _, _ = f_plot(c1)
        f2, _, _ = f_plot(c2)
        f3, _, _ = f_plot(c3)
        f4, _, _ = f_plot(c4)
        obj_path = path[2, :]
        levels = np.linspace(np.amin(obj_path), 2 * np.amax(obj_path))
        min_path_levels = np.amin(levels)
        max_path_levels = np.amax(levels)

        min_f = np.amin(
            np.array([np.squeeze(f1), np.squeeze(f2), np.squeeze(f3), np.squeeze(f4), np.squeeze(min_path_levels)]))
        max_f = np.amax(
            np.array([np.squeeze(f1), np.squeeze(f2), np.squeeze(f3), np.squeeze(f4), np.squeeze(max_path_levels)]))
        int_levels = np.linspace(min_f, max_f, num=30)
        Z = np.zeros(X1.shape)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                x = np.array([[X1[i, j]], [X2[i, j]]])
                Z[i, j] = f_plot(x)[0]
        fig, ax = plt.subplots(figsize=(6,6))
        CS = ax.contour(X1, X2, Z, levels=int_levels)

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        # make a colorbar for the contour lines
        CB = fig.colorbar(CS)
        ax.plot(path[0, :-1], path[1, :-1], 'bo')
        ax.plot(path[0, -1], path[1, -1], 'r*')
        ax.imshow(((X1 >= 1 - X2) & (X1 <= 2) & (X2 >= 0) & (X2 <= 1)).astype(int),
                  extent=(X1.min(), X1.max(), X2.min(), X2.max()), origin="lower", cmap="Greys", alpha=0.3)
        plt.show()
        print(x_s)
        self.assertEqual(True, np.allclose(np.array([[2],[1]]),x_s))

    def test_qp(self):
        x0 = np.array([[0.1],
                       [0.2],
                       [0.7]])
        f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs = examples.constrained_quad_testcase()
        x_s, path = constrained_min.interior_pt(f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)

        # plotting
        plt.figure('test_qp', figsize=(10, 10))
        ax = plt.subplot(111, projection='3d')
        # x+y+z=1 : the feasible region is a triangle in 3d - define its edges
        x1 = np.array([1, 0, 0])
        y1 = np.array([0, 1, 0])
        z1 = np.array([0, 0, 1])
        ax.scatter(x1, y1, z1)

        # create vertices from points
        verts = [list(zip(x1, y1, z1))]
        # create the feasible polygon
        srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
        # add polygon to the figure
        ax.add_collection3d(srf)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.plot(path[1,:], path[0,:], zs=path[2,:])
        ax.scatter(path[1, -1], path[0, -1], zs=path[2, -1],c='r',marker='*')
        ax.view_init(30, 30)
        plt.show()
        self.assertEqual(True, np.allclose(np.array([[0.5], [0.5], [0]]), x_s))


if __name__ == '__main__':
    unittest.main()
