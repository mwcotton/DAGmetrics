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
    """
    TODO: make the function take start and end indices
    """
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

def longest_path(connections):
    """
    TODO: make the function take start and end indices
    """
    n = connections.shape[0]
    dist, prev = djikstre(-1*connections)
    u = n-1

    path=[]
    path.append(u)

    while u != n-2:
        u = prev[u]
        path.append(u)
        
    return path, -1*dist[n-1]


def bi_djikstre(connection_mat):
    """
    Not working. Stopping conditions are incorrect.
    
    Returns dictionaries with;
    -shortest distances to the nodes from the source, [0, 0] (labelled by index).
    -previous node for shortest path from the source, [0, 0] (labelled by index).
    Untested for disconnected graphs (from the source, [0, 0])
    
    Developed to become a bidirectional search.
    """
    n = connection_mat.shape[0]
    
    dist_f, prev_f = {}, {}
    Q_f  = list(range(n))
    
    dist_b, prev_b = {}, {}
    Q_b  = list(range(n))
    
    for i in Q_f:
        dist_f[i] = np.inf
    dist_f[n-2] = 0.0
    
    for i in Q_b:
        dist_b[i] = np.inf
    dist_b[n-1] = 0.0
    
    done_f = []
    done_b = []
    
    while not (set(done_b) & set(done_f)):
        
        for di, dist, prev, Q, done, connections, end in zip(['A', 'B'],[dist_b, dist_f], [prev_b, prev_f], [Q_b, Q_f], [done_b, done_f], [connection_mat.transpose(), connection_mat], [' ','\n']):

            min_dist = min([dist[key] for key in Q])
            u = [key for key in Q if dist[key] == min_dist][0]
#             print(u, di, end=end)

            for v in np.nonzero(connections[:, u])[0]:
#                 print(np.nonzero(connections[:, u])[0])
                alt = dist[u]+connections[v, u]
#                 print(dist)
#                 print(dist[u], alt)

                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
#                     print('added to prev', di, prev)
#                     print('added to dist', di, dist)
                    
            done.append(u)
            Q.remove(u)
                
    meeting_point = list(set(done_b) & set(done_f))[0]
    
#     print('Meeting point:', meeting_point)

    path_b=[]
    path_f=[]

#     path_f.append(u)
    
    u = meeting_point
    
    while u != n-1:
#         print(u)
        u = prev_b[u]
        path_b.append(u)
        
    u = meeting_point

    while u != n-2:
#         print(u)
        u = prev_f[u]
        path_f.append(u)
    
    full_path =path_b[::-1]
    full_path.append(meeting_point)
    full_path.extend(path_f)
    
    return full_path

# def get_connections_ND(points, radius=.1, pval=2):
#     """
#     Finds all the connections between the points given. Connections must satisfy:
#     -box square direction (i.e. all coords in _to_ greater than in _from_) - here generalise to N coords
#     -distance separated nodes less that 'radius' (according to Minkowski distance with p='pval')
#     Resulting graph is directed and acyclic

#     N taken from the dimension of points.
#     """

#     radp = radius**pval

#     x_rows, x_cols = np.meshgrid(points[0, :], points[0, :])
#     y_rows, y_cols = np.meshgrid(points[1, :], points[1, :])
#     x_diffs = (x_cols - x_rows)
#     y_diffs =  (y_cols - y_rows)
#     distsp = (x_diffs**pval + y_diffs**pval)*(x_diffs>0)*(y_diffs>0)
#     connections = (((distsp<radp)*distsp)**(1/pval))*(x_diffs>0)*(y_diffs>0)

#     return(np.nan_to_num(connections))

def get_connections_ND(points, radius=.1, pval=2):
    """
    Finds all the connections between the points given. Connections must satisfy:
    -box square direction (i.e. all coords in _to_ greater than in _from_) - here generalise to N coords
    -distance separated nodes less that 'radius' (according to Minkowski distance with p='pval')
    Resulting graph is directed and acyclic

    N taken from the dimension of points.
    """
    N = points.shape[0]

    radp = radius**pval

    meshed = [np.meshgrid(points[i, :], points[i, :]) for i in range(N)]
    diffs = np.array([cols-rows for rows, cols in meshed])
    box_cube_condition = (diffs > 0).all(axis=0)
    distsp = (diffs**pval).sum(axis=0)

    connections = (((distsp<radp)*distsp)**(1/pval))*box_cube_condition

    return(np.nan_to_num(connections))


def box_counting(points, dists={}, samples=100, connections=[], radius=[], pval=[], weighted=False):

    if not (connections or dists):
        print('Getting connections')
        connections = mt.get_connections_ND(points, radius, pval)

    if not (weighted or dists):
        connections = connections.astype(bool).astype(int)
        print('Not weighted')
            
    if not dists:
        print('Getting dists')
        dists, _ = mt.djikstre(connections)
        
    results = []
    options = [key for key in dists.keys() if np.isfinite(dists[key])]

    for _ in range(samples):
        sample_point = np.random.choice(options)
        options.remove(sample_point)

        count = np.sum((points[1, :] < points[1, sample_point])*(points[0, :] < points[0, sample_point]))
        
        results.append([dists[sample_point], count])
        
    return np.array(results).transpose()