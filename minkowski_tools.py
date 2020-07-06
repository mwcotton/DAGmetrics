import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt

def djikstre(connection_mat):
    """
    Returns dictionaries with;
    -shortest distances to the nodes from the source, [0, 0] (labelled by index).
    -previous node for shortest path from the source, [0, 0] (labelled by index).
    Untested for disconnected graphs (from the source, [0, 0])
    """
    n = connection_mat.shape[0]
    dist, prev = {}, {}
    Q = list(range(n))
    
    for i in Q:
        dist[i] = np.inf
    dist[n-2] = 0.0
    
    while(len(Q)>0):

        min_dist = min([dist[key] for key in Q])
        u = [key for key in Q if dist[key] == min_dist][0]
        Q.remove(u)

        for v in np.nonzero(connection_mat[:, u])[0]:
            
            alt = dist[u]+connection_mat[v, u]
            
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                
    return dist, prev


def get_connections(points, radius=.1, pval=2):
    """
    Finds all the connections between the points given. Connections must satisfy:
    -box square direction (i.e. all coords in _to_ greater than in _from_)
    -distance separated nodes less that 'radius' (according to Minkowski distance with p='pval')
    Resulting graph is directed and acyclic
    """
    radp = radius**pval

    x_rows, x_cols = np.meshgrid(points[0, :], points[0, :])
    y_rows, y_cols = np.meshgrid(points[1, :], points[1, :])
    x_diffs = (x_cols - x_rows)
    y_diffs =  (y_cols - y_rows)
    distsp = (x_diffs**pval + y_diffs**pval)*(x_diffs>0)*(y_diffs>0)
    connections = (((distsp<radp)*distsp)**(1/pval))*(x_diffs>0)*(y_diffs>0)

    return(np.nan_to_num(connections))


def plot_points_simple(ax, points, paths=[], path_labels=[]):
    """
    Basic plotting function to visaulise points and paths.
    Plots on the 'ax' axis.
    """
    cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    ax.scatter(*points, alpha=0.1, c='k')
#     add functionality to ignore labels
    for path, label, col in zip(paths, path_labels, cols):
        path_points = np.array([points[:, u] for u in path]).transpose()
        ax.plot(*path_points, alpha=.8,label=label, c=col)
        ax.scatter(*path_points, c=col, alpha=0.6)
        
    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.set_axis_off()
    
    if path_labels:
        ax.legend()

    return ax


def plot_path_points(ax, points=[], paths=[], path_labels=[]):
    """
    Basic plotting function to visaulise paths, without requiring the points.
    Plots on the 'ax' axis.
    """
    cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # if points:
    #     ax.scatter(*points, alpha=0.1, c='k')

#     add functionality to ignore labels
    for path, label, col in zip(paths, path_labels, cols):
        ax.plot(*path, alpha=.8, c=col, label=label)
        ax.scatter(*path, alpha=.6, c=col)
        
    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.set_axis_off()
    
    if path_labels:
        ax.legend()

    return ax

def shortest_path(connections):

    n = connections.shape[0]
    dist, prev = djikstre(connections)
    u = n-1

    path=[]
    path.append(u)

    while u != n-2:
        u = prev[u]
        path.append(u)
        
    return path, dist[n-1]

def points_between(points, path1, path2=np.array([[0, 1], [0, 1]])):
    """
    Returns the coordinates and boolean indices for points between the two paths.
    Takes the paths as points
    """
    f1, f2 = sp.interpolate.interp1d(*path1), sp.interpolate.interp1d(*path2)

    above_path1 = points[1, :] < f1(points[0, :])
    above_path2 = points[1, :] < f2(points[0, :])

    return np.logical_xor(above_path1, above_path2)

def path_angles(path):
    """
    """
    dxs = path[0, :-1]-path[0, 1:]
    dys = path[1, :-1]-path[1, 1:]
    path_angles = np.arctan(dys/dxs)
    return path_angles