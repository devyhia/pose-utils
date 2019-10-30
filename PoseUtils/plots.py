"""
Debugging camera pose localization trajectories is difficult without proper visualization tools.
Most of the time, we dump the trajectories and visualize them in Matlab.
However, in my experience, that results in a very bad user experience.
Therefore, I will be sharing here a few visualization utility methods that I found helpful in my research.
"""
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plotTrajectories(trajectory_list, legends=None, plot_start=False, plot_end=False):
    """
    Plots a 3D visualization of multiple trajectories.
    Trajectors are represented as a sequence of transformation matrices.

    Args:
        trajectory_list: a list of transformation matrices.
        legends: a list of labels for each trajectory.
        plot_start: boolean. Decides whether or not to plot a dot indicating the start of the trajectory.
        plot_end: boolean. Decides whether or not to plot a dot indicating the end of the trajectory.

    Example:
        ```
        >> plotTrajectories([trajectory], legends=["Trajectory 1"], plot_start=True)
        ```

        .. image:: /PoseUtils/imgs/example-trajectory.png
    """

    fig = plt.figure(figsize=(7, 7))
    ax = p3.Axes3D(fig)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for trajectory in trajectory_list:
        print(len(trajectory))
        frames = np.array([tr[:3, -1] for tr in trajectory])

        ax.plot(frames[:, 0], frames[:, 1], frames[:, 2])
        if plot_start:
            start = frames[0, :]
            ax.scatter(start[0], start[1], start[2], s=35, c="red")

        if plot_end:
            end = frames[-1, :]
            ax.scatter(end[0], end[1], end[2], s=35, c="yellow")

    if legends:
        ax.legend(legends)

    return ax


def plotAxis(
    translation,
    rotation,
    label=None,
    ax=None,
    figsize=(5, 5),
    point_color="black",
    normalize_translation=False,
    cameraSize=1,
):
    """
    Plots a camera's coordinates in 3D.

    Args:
        translation: a 3x1 translation vector.
        rotation: a 3x3 rotation matrix.
        label: string; a label for the trajectory to be drawn.
        ax: Axes3D instance; if this drawing is part of a bigger drawing function, this share the parent's axis (so that more than function could draw on the same canvas).
        figsize: Tuple[int, int]; If new axis is created, this instructs the figure size to the function-- i.e. `figsize = (5,5)`.
        point_color: string; This decides the color of the camera center.
        normalize_translation: boolean; Whether or not to normalize the the camera center (i.e. the translation passed to this function).
    """
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = p3.Axes3D(fig)

    # Setting the axes properties
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    r1 = rotation[:, 0] / np.linalg.norm(rotation[:, 0])
    r2 = rotation[:, 1] / np.linalg.norm(rotation[:, 1])
    r3 = rotation[:, 2] / np.linalg.norm(rotation[:, 2])

    if normalize_translation:
        translation = translation / np.linalg.norm(translation)

    d1 = np.array([translation, (translation + cameraSize * r1)])
    d2 = np.array([translation, (translation + cameraSize * r2)])
    d3 = np.array([translation, (translation + cameraSize * r3)])

    d_center = np.array([d1, d2, d3]).mean(axis=0)

    # Plot center
    ax.scatter(translation[0], translation[1], translation[2], c=point_color, s=35)

    # Plot axes
    ax.plot(d1[:, 0], d1[:, 1], d1[:, 2], color="red")
    ax.plot(d2[:, 0], d2[:, 1], d2[:, 2], color="green")
    ax.plot(d3[:, 0], d3[:, 1], d3[:, 2], color="blue")

    ax.plot(d_center[:, 0], d_center[:, 1], d_center[:, 2], color="black")

    # Plot triangles to show camera direction
    face_vertices = [d1[1], d2[1], d3[1]]
    verts = [face_vertices]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.1))

    if label:
        ax.text(
            translation[0],
            translation[1],
            translation[2],
            label,
            size=20,
            zorder=1,
            color="k",
        )

    return ax


def plotAxisMany(poses, labels=None, figsize=(5, 5), **kwargs):
    """
    Plots many camera posees on the same axis in 3D.

    Args:
        poses: A trajectory-- i.e. a list of transformation matrices (each is 4x4 matrix). 
        labels: A list of labels; Total labels should match the number of camera poses. 
        figsize: Tuple[int, int]; The figure the size of the plot.

    Example:
        This method produces the following plot.
        .. image:: /PoseUtils/imgs/camera-plots.png
    """
    ax = None

    if not labels:
        labels = [None] * len(poses)

    for tr, label in zip(poses, labels):
        r = tr[:3, :3]
        t = tr[:3, -1]
        ax = plotAxis(
            t, r, label=label, point_color="black", ax=ax, figsize=figsize, **kwargs
        )

    return ax
