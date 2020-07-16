import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt

# pvals =  np.arange(0.5, 1.55, 0.05)
# results = [0.166667, 0.205707, 0.244451, 0.282227, 0.318605, 0.353328, 0.386266, 0.417373, 0.446661, 0.474179, 0.5, 0.524209, 0.546899, 0.568163, 0.588095, 0.606786, 0.624321, 0.640783, 0.656247, 0.670785, 0.684463]

pvals = np.arange(0.1, 2.01, 0.01)
results = [5.41254e-6, 0.0000182211, 0.0000499262, 0.000116795, 0.000241366, \
0.000451785, 0.000780399, 0.00126194, 0.00193161, 0.00282332, \
0.00396825, 0.00539377, 0.00712273, 0.00917313, 0.011558, 0.0142857, \
0.01736, 0.0207805, 0.0245433, 0.0286415, 0.0330654, 0.0378032, \
0.0428413, 0.0481648, 0.0537579, 0.0596041, 0.0656865, 0.0719881, \
0.0784918, 0.0851809, 0.0920388, 0.0990496, 0.106198, 0.113468, \
0.120846, 0.128319, 0.135873, 0.143496, 0.151177, 0.158904, 0.166667, \
0.174456, 0.182263, 0.190079, 0.197896, 0.205707, 0.213505, 0.221284, \
0.229038, 0.236762, 0.244451, 0.252101, 0.259707, 0.267265, 0.274773, \
0.282227, 0.289625, 0.296964, 0.304241, 0.311455, 0.318605, 0.325687, \
0.332702, 0.339648, 0.346523, 0.353328, 0.360061, 0.366721, 0.373309, \
0.379824, 0.386266, 0.392634, 0.398928, 0.40515, 0.411298, 0.417373, \
0.423375, 0.429304, 0.435161, 0.440947, 0.446661, 0.452304, 0.457877, \
0.46338, 0.468814, 0.474179, 0.479476, 0.484707, 0.48987, 0.494968, \
0.5, 0.504968, 0.509872, 0.514713, 0.519492, 0.524209, 0.528866, \
0.533462, 0.537999, 0.542478, 0.546899, 0.551262, 0.55557, 0.559822, \
0.56402, 0.568163, 0.572253, 0.57629, 0.580276, 0.584211, 0.588095, \
0.59193, 0.595715, 0.599453, 0.603143, 0.606786, 0.610382, 0.613934, \
0.61744, 0.620902, 0.624321, 0.627697, 0.63103, 0.634322, 0.637572, \
0.640783, 0.643953, 0.647084, 0.650176, 0.65323, 0.656247, 0.659226, \
0.662169, 0.665076, 0.667948, 0.670785, 0.673587, 0.676356, 0.679091, \
0.681793, 0.684463, 0.687101, 0.689708, 0.692284, 0.694829, 0.697344, \
0.699829, 0.702285, 0.704712, 0.707111, 0.709482, 0.711826, 0.714142, \
0.716431, 0.718694, 0.720931, 0.723143, 0.725329, 0.727491, 0.729627, \
0.73174, 0.733829, 0.735894, 0.737936, 0.739955, 0.741952, 0.743927, \
0.745879, 0.74781, 0.74972, 0.751609, 0.753477, 0.755325, 0.757153, \
0.75896, 0.760749, 0.762518, 0.764267, 0.765999, 0.767711, 0.769406, \
0.771082, 0.772741, 0.774382, 0.776005, 0.777612, 0.779202, 0.780776, \
0.782332, 0.783873, 0.785398]

headers = 'p', 'r', 'n', 'short_lengthBool', 'long_lengthBool', 'short_length', 'long_length', 'short_pathBoolpoints', 'long_pathBoolpoints', 'short_pathpoints', 'long_pathpoints'

# x = np.arange(0.1, 2.01, 0.01)

kernel_area2D = scipy.interpolate.CubicSpline(pvals, results)

def norm_kernel_2D(pval, area):
    """
    Returns the 'radius' required to make a kernel descirbed by pval have the desired input area
    (Interpolated between 0.1 and 2, therefore vals outside this may have large uncertainties).
    """
    area_r_1 = kernel_area2D(pval)
    r = np.sqrt(area/area_r_1)
    return (r)

# if __name__ == "__main__":
#     # this won't be run when imported

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
    
    if prev.get(u):
        while u != n-2:
            u = prev[u]
            path.append(u)
            
        return path, dist[n-1]

    else:
        return [], 0.0
    

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

    if prev.get(u):
        while u != n-2:
            u = prev[u]
            path.append(u)
            
        return path, -1*dist[n-1]

    else:
        return [], 0.0


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

def points_str(string):

    clean = string.replace(']', '').replace(' ', '').replace('[', '')
    coords_flat = np.array(clean.split(','), dtype=float)
    
    return coords_flat.reshape(-1, 2).transpose()

def separate_simulations(ns, ps, rs, savename, verbose=True):
    """
    savename is a text file
    """
    if verbose:

        for n, p, r in zip(ns, ps, rs):
            
            print('n-{}, p-{}, r-{}'.format(n, p, r))
            
            rand_points = np.random.uniform(size=(2, n-2))
            edge_points = np.array([[0.0, 1.0],[0.0, 1.0]])
            points = np.concatenate((rand_points, edge_points), axis=1)
            
            print('Generated points.', end = '')
            
            connections = get_connections(points, pval=p, radius=r)

            print('Got connections.')
            
            print('Getting paths.', end = '...')
            
            print('Longest.', end = '...')
            long_pathBool, long_lengthBool = longest_path(connections.astype(bool))
            long_path, long_length = longest_path(connections)

            print('Shortest.', end = '...')
            short_pathBool, short_lengthBool = shortest_path(connections.astype(bool))
            short_path, short_length = shortest_path(connections)

            short_pathBoolpoints, long_pathBoolpoints, short_pathpoints, long_pathpoints = [[list(points[:, u]) for u in indexes] for indexes in [short_pathBool, long_pathBool, short_path, long_path]]

            print('Saving file -> ' + savename)
            file1 = open(savename,"a") 

            file1.writelines('{} - {} - {} - {} - {} - {} - {} - {} - {} - {} - {}\n'.format(p, r, n, short_lengthBool, long_lengthBool, short_length, long_length, short_pathBoolpoints, long_pathBoolpoints, short_pathpoints, long_pathpoints))
            file1.close()
            
    else:

        for n, p, s in zip(ns, ps, rs):

            rand_points = np.random.uniform(size=(2, n-2))
            edge_points = np.array([[0.0, 1.0],[0.0, 1.0]])
            points = np.concatenate((rand_points, edge_points), axis=1)

            connections = get_connections(points, pval=p, radius=r)

            long_pathBool, long_lengthBool = longest_path(connections.astype(bool))
            long_path, long_length = longest_path(connections)

            short_pathBool, short_lengthBool = shortest_path(connections.astype(bool))
            short_path, short_length = shortest_path(connections)

            short_pathBoolpoints, long_pathBoolpoints, short_pathpoints, long_pathpoints = [[list(points[:, u]) for u in indexes] for indexes in [short_pathBool, long_pathBool, short_path, long_path]]

            file1 = open(savename,"a") 

            file1.writelines('{} - {} - {} - {} - {} - {} - {} - {} - {} - {} - {}\n'.format(p, r, n, short_lengthBool, long_lengthBool, short_length, long_length, short_pathBoolpoints, long_pathBoolpoints, short_pathpoints, long_pathpoints))
            file1.close()
        
    return True