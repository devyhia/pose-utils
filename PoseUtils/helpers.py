"""
Helpers functions to solve common pose-related problems-- i.e. converting from left-handed to right-handed coordinate system for example.
"""

import numpy as np


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
    eps = 1e-8

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
