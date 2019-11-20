"""
Helpers functions to solve common pose-related problems-- i.e. converting from left-handed to right-handed coordinate system for example.
"""

import numpy as np
from numpy import sqrt, ones, array
from numpy.linalg import solve


def line_from_points(p1, p2):
    """
    Stacks two points to make a line.

    Args:
        p1: 1x2 or 2x1 numpy array with two elements.
        p2: 1x2 or 2x1 numpy array with two elements.
    """
    return np.vstack([p1.flatten(), p2.flatten()])


def flip_coordinate_system(T: np.ndarray, axis=2) -> np.ndarray:
    """
    Flips between coordinate system convetions (right-handed and left-handed).
    In order to flip from right-handed to left-handed coordinate system (or vice-versa), we need to invert one of the axis or all of the axis.
    Here we only flip one of the axis; By default, we opt out to invert te Z-Axis (i.e. `axis=2`).

    Args:
        T: a 4x4 transformation matrix of type `numpy.ndarray`.
        axis: int (defaut = 2). The axis around which we do the flip; `axis` could be 0, 1 or 2.

    Output:
        A 4x4 transformation matrix of type `numpy.ndarray`.
    """
    return T[:, axis] * -1


def line_line_intersection(p1, p2, p3, p4) -> np.ndarray:
    """
    Line-Line Intersection Function.
    Calculates the line segment pa_pb that is the shortest route
    between two lines p1_p2 and p3_p4. Calculates also the values of
    mua and mub where 
         pa = p1 + mua (p2 - p1) 
         pb = p3 + mub (p4 - p3)

    Args:
        p1: Starting point of first line.
        p2: Ending point of first line.
        p3: Starting piont of second line.
        p4: Ending point of second line.

    Exception:
        Raises exception if no such intersection is possible.

    Reference:
        This a simple conversion to Python from MATLAB code given by Cristian Dima
        (csd@cmu.edu). The original method is a translation of the C code by Paul
        Bourke at http://astronomy.swin.edu.au/~pbourke/geometry/lineline3d/.
    """
    p13 = p1 - p3
    p43 = p4 - p3
    eps = 2.2204e-16  # This number is quoted from MATLAB "eps".

    if np.all(p43 < eps):
        raise Exception("Could not find an intersection")

    p21 = p2 - p1

    if np.all(p21 < eps):
        raise Exception("Could not find an intersection")

    d1343 = p13[0] * p43[0] + p13[1] * p43[1] + p13[2] * p43[2]
    d4321 = p43[0] * p21[0] + p43[1] * p21[1] + p43[2] * p21[2]
    d1321 = p13[0] * p21[0] + p13[1] * p21[1] + p13[2] * p21[2]
    d4343 = p43[0] * p43[0] + p43[1] * p43[1] + p43[2] * p43[2]
    d2121 = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2]

    denom = d2121 * d4343 - d4321 * d4321

    if denom < eps:
        raise Exception("Could not intersect the lines")

    numer = d1343 * d4321 - d1321 * d4343

    mua = numer / denom
    mub = (d1343 + d4321 * mua) / d4343

    pa = p1 + mua * p21
    pb = p3 + mub * p43

    return (pa + pb) / 2


def intersection_point_of_3d_lines(PA: np.ndarray, PB: np.ndarray) -> np.ndarray:
    """
    Find intersection points closet to all given lines (in a least squares sense).

    This implementation is a direct translation of the Matlab implementation provided by: https://de.mathworks.com/matlabcentral/fileexchange/37192-intersection-point-of-lines-in-3d-space

    Args:
        PA: Nx3 Numpy array. Starting points for each line.
        PB: Nx3 Numpy array. Ending points for each line.

    Returns:
        3x1 Numpy Array. Intersection point for all lines.

    Example:
        The following figures illustrate 13 rays, each passing by (or around) the origin point (i.e. `[0,0,0]`).

        ### No Error Example
        In this image, we have 13 lines, all of which are crossing the origin. Our expectation is that the returned intersection point should be the origin.
        .. image:: imgs/intersection-no-error.png

        ### Guassian Noise Example
        When adding gaussian noise to start and end points of the line segments (that describe the rays), the function is still fairly resistent to the noise.
        In the following example, we added noise with a zero mean and a 0.1 standard deviation.

        .. image:: imgs/intersection-tiny-error.png

        We further added more gaussian nose with zero mean and 0.5 standard deviation to obtain the following:

        .. image:: imgs/intersection-small-error.png

        In all three cases, the function returns the origin as the intersection point.
    """
    Si = PB - PA  # N lines described as vectors
    Si_Norm = np.sqrt(np.sum(Si ** 2, axis=1))
    ni = Si / Si_Norm[:, np.newaxis]  # Normalize vectors (each)
    nx = ni[:, 0]
    ny = ni[:, 1]
    nz = ni[:, 2]

    SXX = sum(nx ** 2 - 1)
    SYY = sum(ny ** 2 - 1)
    SZZ = sum(nz ** 2 - 1)
    SXY = sum(nx * ny)
    SXZ = sum(nx * nz)
    SYZ = sum(ny * nz)
    S = array([[SXX, SXY, SXZ], [SXY, SYY, SYZ], [SXZ, SYZ, SZZ]])
    CX = sum(PA[:, 0] * (nx ** 2 - 1) + PA[:, 1] * (nx * ny) + PA[:, 2] * (nx * nz))
    CY = sum(PA[:, 0] * (nx * ny) + PA[:, 1] * (ny ** 2 - 1) + PA[:, 2] * (ny * nz))
    CZ = sum(PA[:, 0] * (nx * nz) + PA[:, 1] * (ny * nz) + PA[:, 2] * (nz ** 2 - 1))
    C = array([CX, CY, CZ])
    P_intersect = solve(S, C).T

    return P_intersect
