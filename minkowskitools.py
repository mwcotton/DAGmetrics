import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt

pvals = np.concatenate((np.arange(0.1, 2.01, 0.01), np.arange(2.1, 5.01, 0.1)))

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
0.782332, 0.783873, 0.785398, 0.799815, 0.812852, 0.824675, 0.835428, 0.845234, \
0.854198, 0.862414, 0.869961, 0.876909, 0.883319, 0.889245, 0.894734, \
0.899827, 0.904562, 0.90897, 0.913082, 0.916922, 0.920515, 0.92388, \
0.927037, 0.930003, 0.932792, 0.935418, 0.937893, 0.94023, 0.942437, \
0.944524, 0.946501, 0.948374, 0.95015]

# pvals =  np.arange(0.5, 1.55, 0.05)
# results = [0.166667, 0.205707, 0.244451, 0.282227, 0.318605, 0.353328, 0.386266, 0.417373, 0.446661, 0.474179, 0.5, 0.524209, 0.546899, 0.568163, 0.588095, 0.606786, 0.624321, 0.640783, 0.656247, 0.670785, 0.684463]

# pvals = np.arange(0.1, 2.01, 0.01)
# results = [5.41254e-6, 0.0000182211, 0.0000499262, 0.000116795, 0.000241366, \
# 0.000451785, 0.000780399, 0.00126194, 0.00193161, 0.00282332, \
# 0.00396825, 0.00539377, 0.00712273, 0.00917313, 0.011558, 0.0142857, \
# 0.01736, 0.0207805, 0.0245433, 0.0286415, 0.0330654, 0.0378032, \
# 0.0428413, 0.0481648, 0.0537579, 0.0596041, 0.0656865, 0.0719881, \
# 0.0784918, 0.0851809, 0.0920388, 0.0990496, 0.106198, 0.113468, \
# 0.120846, 0.128319, 0.135873, 0.143496, 0.151177, 0.158904, 0.166667, \
# 0.174456, 0.182263, 0.190079, 0.197896, 0.205707, 0.213505, 0.221284, \
# 0.229038, 0.236762, 0.244451, 0.252101, 0.259707, 0.267265, 0.274773, \
# 0.282227, 0.289625, 0.296964, 0.304241, 0.311455, 0.318605, 0.325687, \
# 0.332702, 0.339648, 0.346523, 0.353328, 0.360061, 0.366721, 0.373309, \
# 0.379824, 0.386266, 0.392634, 0.398928, 0.40515, 0.411298, 0.417373, \
# 0.423375, 0.429304, 0.435161, 0.440947, 0.446661, 0.452304, 0.457877, \
# 0.46338, 0.468814, 0.474179, 0.479476, 0.484707, 0.48987, 0.494968, \
# 0.5, 0.504968, 0.509872, 0.514713, 0.519492, 0.524209, 0.528866, \
# 0.533462, 0.537999, 0.542478, 0.546899, 0.551262, 0.55557, 0.559822, \
# 0.56402, 0.568163, 0.572253, 0.57629, 0.580276, 0.584211, 0.588095, \
# 0.59193, 0.595715, 0.599453, 0.603143, 0.606786, 0.610382, 0.613934, \
# 0.61744, 0.620902, 0.624321, 0.627697, 0.63103, 0.634322, 0.637572, \
# 0.640783, 0.643953, 0.647084, 0.650176, 0.65323, 0.656247, 0.659226, \
# 0.662169, 0.665076, 0.667948, 0.670785, 0.673587, 0.676356, 0.679091, \
# 0.681793, 0.684463, 0.687101, 0.689708, 0.692284, 0.694829, 0.697344, \
# 0.699829, 0.702285, 0.704712, 0.707111, 0.709482, 0.711826, 0.714142, \
# 0.716431, 0.718694, 0.720931, 0.723143, 0.725329, 0.727491, 0.729627, \
# 0.73174, 0.733829, 0.735894, 0.737936, 0.739955, 0.741952, 0.743927, \
# 0.745879, 0.74781, 0.74972, 0.751609, 0.753477, 0.755325, 0.757153, \
# 0.75896, 0.760749, 0.762518, 0.764267, 0.765999, 0.767711, 0.769406, \
# 0.771082, 0.772741, 0.774382, 0.776005, 0.777612, 0.779202, 0.780776, \
# 0.782332, 0.783873, 0.785398]

# Table[Integrate[(1 - x^p)^(1/p), {x, 0, 1}], {p, 2, 5, 0.1} ]
# {0.785398, 0.799815, 0.812852, 0.824675, 0.835428, 0.845234, \
# 0.854198, 0.862414, 0.869961, 0.876909, 0.883319, 0.889245, 0.894734, \
# 0.899827, 0.904562, 0.90897, 0.913082, 0.916922, 0.920515, 0.92388, \
# 0.927037, 0.930003, 0.932792, 0.935418, 0.937893, 0.94023, 0.942437, \
# 0.944524, 0.946501, 0.948374, 0.95015}

headers = 'p', 'r', 'n', 'short_lengthBool', 'long_lengthBool', 'short_length', 'long_length', 'short_pathBoolpoints', 'long_pathBoolpoints', 'short_pathpoints', 'long_pathpoints'
all_paths = 'p', 'r', 'n', 'short_met', 'long_met', 'short_net', 'long_net', 'lMin_F', 'lMax_F', 'lMin_B', 'lMax_B'
# x = np.arange(0.1, 2.01, 0.01)

kernel_area2D = scipy.interpolate.CubicSpline(pvals, results)

def r1_area2D(pval):
    if pval < 7.45:
        return kernel_area2D(pval)
    else:
        return (1-1/(pval*(pval+1))) #approximation of the integral for 

def norm_kernel_2D(pval, area):
    """
    Returns the 'radius' required to make a kernel descirbed by pval have the desired input area
    (Interpolated between 0.1 and 2, therefore vals outside this may have large uncertainties).
    """
    area_r_1 = kernel_area2D(pval)
    r = np.sqrt(area/area_r_1)
    return (r)

    # TODO: find as asymtopic functional form for this as pval gets very large (should tend to 1)

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

def shortest_path_old(connections):
    """
    TODO: make the function take start and end indices
    """
    n = connections.shape[0]
    dist, prev = djikstre(connections)
    u = n-1

    path=[]
    path.append(u)
    
    if prev.get(u) is not None:
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

def longest_path_old(connections):
    """
    TODO: make the function take start and end indices
    """
    n = connections.shape[0]
    dist, prev = djikstre(-1*connections)
    u = n-1

    path=[]
    path.append(u)

    if prev.get(u) is not None:
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

    TODO: finish
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
        connections = get_connections_ND(points, radius, pval)

    if not (weighted or dists):
        connections = connections.astype(bool).astype(int)
        print('Not weighted')
            
    if not dists:
        print('Getting dists')
        dists, _ = djikstre(connections)
        
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
    #TODO: find bug that occasionally does not find the shortest path
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


def longest_path(connection_mat):
    """
    TODO: make the function take start and end indices
    Also: ensure that this works with the overall
    """
    
    n = connection_mat.shape[0]
    dist, prev = {}, {}
    
    for i in range(n):
        dist[i] = -1*np.inf
    dist[n-2] = 0
        
    for u in topo_sort(connection_mat):
        for v in np.nonzero(connection_mat[:, u])[0]:
            
            alt = dist[u]+connection_mat[v, u]
            
            if alt > dist[v]:
                dist[v] = alt
                prev[v] = u
   
    u = n-1

    path=[]
    path.append(u)

    if prev.get(u) is None:
        return [], 0

    if prev.get(u) is not None:
        while u != n-2:
            u = prev[u]
            path.append(u)
            
    return path, dist[n-1]

def shortest_path(connection_mat):
    """
    TODO: make the function take start and end indices
    """

    n = connection_mat.shape[0]
    dist, prev = {}, {}
    
    for i in range(n):
        dist[i] = np.inf
    dist[n-2] = 0
        
    for u in topo_sort(connection_mat):
        for v in np.nonzero(connection_mat[:, u])[0]:
            
            alt = dist[u]+connection_mat[v, u]
            
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
   
    u = n-1
    
    path=[]
    path.append(u)

    if prev.get(u) is not None:
        while u != n-2:
            u = prev[u]
            path.append(u)
            
        return path, dist[n-1]

    else:
        return [], 0.0

def longest_path_with_ponits(connection_mat, points):
    """
    TODO: make the function take start and end indices
    Also: ensure that this works with the overall
    """
    dist, prev = {}, {}
    
    for i in range(n):
        dist[i] = -1*np.inf
    dist[n-2] = 0
        
    for u in np.lexsort((points[0], points[1])):
        for v in np.nonzero(connection_mat[:, u])[0]:
            
            alt = dist[u]+connection_mat[v, u]
            
            if alt > dist[v]:
                dist[v] = alt
                prev[v] = u
   
    u = n-1

    path=[]
    path.append(u)

    if prev.get(u) is not None:
        while u != n-2:
            u = prev[u]
            path.append(u)
            
    return path, dist[n-1]


def topo_sort(connections):
    
    connection_mat = connections.copy()
    n = len(connection_mat)
    Q = list(range(n))
    
    order = []    
    
    while(len(Q)>0):
        no_entry = np.where(np.sum(connection_mat, axis=1) == 0)[0]
        for u in list(set(Q).intersection(set(no_entry))):
            order.append(u)
            connection_mat[:, u] =[0]*n
            Q.remove(u)
            
    return order

#     DAG PATH ALOG: algorithm dag-longest-path is input: Directed acyclic graph G output: Length of the longest path
# length_to = array with |V(G)| elements of type int with default value 0

# for each vertex v in topOrder(G) do
#     for each edge (v, w) in E(G) do
#         if length_to[w] <= length_to[v] + weight(G,(v, w)) then
#             length_to[w] = length_to[v] + weight(G,(v, w))

# return max(length_to[v] for v in V(G))

def smallest_r(points,  pval):
    """
    Find the smallest radius for the connections kernel required for the points to percolate for Minkowski pval.
    """

    N = points.shape[0]
    n = points.shape[1]

    meshed = [np.meshgrid(points[i, :], points[i, :]) for i in range(N)]
    diffs = np.array([cols-rows for rows, cols in meshed])
    box_cube_condition = (diffs > 0).all(axis=0)
    distsp = (diffs**pval).sum(axis=0)

    nolimit_connections = (distsp**(1/pval))*box_cube_condition
    nolimit_connections = np.nan_to_num(nolimit_connections)
    
    maxes, prev = {}, {}

    for i in range(n):
        maxes[i] = np.inf
    maxes[n-2] = 0
    
    for u in topo_sort(nolimit_connections): #can replace the top sort with points sorting
        for v in np.nonzero(nolimit_connections[:, u])[0]:

            alt = max(maxes[u], nolimit_connections[v, u])
                        
            if alt < maxes[v]:

                maxes[v] = alt
                prev[v] = u

    u = n-1

    path=[]
    path.append(u)
    
    if prev.get(u) is not None:
            while u != n-2:
                u = prev[u]
                path.append(u)

            return path, maxes[n-1]

    else:
            return [], 0.0

def separate_perc_r(ns, ps, savename, repeats=1):

    for n, pval in zip(ns, ps):
        for _ in range(repeats):
            rand_points = np.random.uniform(size=(2, n-2))
            edge_points = np.array([[0.0, 1.0],[0.0, 1.0]])
            points = np.concatenate((rand_points, edge_points), axis=1)

            min_path, r_min = smallest_r(points,  pval)
            path_points = [list(points[:, u]) for u in min_path]

            file1 = open(savename, "a") 

            file1.writelines('{} - {} - {} -{}\n'.format(pval, n, r_min, path_points))
            file1.close()

    return True

def perc_thresh_n(connections):

    n = len(connections)
    previous = {}

    for ind in [n-2, n-1]:
        previous[ind] = {ind}

    if connections[n-1, n-2]:
        previous[n-1].add(n-2)

    no_points = 2

    
    while (n-2) not in previous[n-1] and no_points<n:
        
        no_points += 1
        
        upto = n-no_points
        previous[upto] = {upto}

        for node_to_current in np.nonzero(connections[n-no_points, n-no_points:])[0]:
            previous[upto].update(previous[upto + node_to_current])

        nodes_from = set(upto + np.nonzero(connections[upto:, upto])[0])

        for other_node in range(upto, n):
            if previous[other_node].intersection(nodes_from):
                previous[other_node].update(previous[upto])

    return no_points

def separate_perc_n(p, r, n_max=None):

    if n_max==None:
        n_max=int(3/(r1_area2D(p)*r*r))

    rand_points = np.random.uniform(size=(2, n_max-2))
    edge_points = np.array([[0.0, 1.0],[0.0, 1.0]])
    points = np.concatenate((rand_points, edge_points), axis=1)

    connections = get_connections(points, radius=r, pval=p)

    return perc_thresh_n(connections)


def ensemble_perc_n(fileName, ps, rs, repeats=1, verbose=True):

    for p, r in zip(ps, rs):
        
        if verbose:
            print(f'p:{p}, r:{r}')
            
        for i in range(repeats):
            if verbose:
                print(i, end=' ')
                
            thresh = separate_perc_n(p, r)
            file1 = open("{}".format(fileName),"a") 
            file1.writelines(f'{p} - {r} - {thresh}\n')
            file1.close()

        if verbose:
            print()
            
    return fileName


def greedy_path(connections, select_func=np.argmax, sink=None, source=None, included=None):
    """
    example of a select function:
    def rand_select(arr):
        return np.random.choice(np.where(arr)[0])
    """
    if included is None:
        included = relevant_points(connections, sink)

    n = len(connections)
    
    if sink is None:
        sink = n-1
    
    if source is None:
        source = n-2

    if included[source] == 0:
        return [], 0

    node = source

    path_length = 0
    path = []
    path.append(node)

    while node != sink:
        
        options = connections[:, node]*included
        
        new_node = select_func(np.ma.masked_where(options == 0, options, copy=False))

        path.append(new_node)
        path_length += connections[new_node, node]

        node = new_node

    return path, path_length


def relevant_points(connections, sink=None):
    
    n = len(connections)
    
    if sink is None:
        sink = n-1
    
    next_time = np.zeros(shape=n, dtype=int)
    next_time[sink] = 1
    
    included = np.zeros(shape=n, dtype=int)
    not_included = np.ones(shape=n, dtype=int)

    while(np.sum(next_time)):
        
        included += next_time
        not_included -= next_time

        next_time = connections[np.where(next_time)].sum(axis=0).astype(bool).astype(int)*not_included

    return included

def demonstrate(points, connections, ax=None):
    """
    Quick function to help visualising (mainly for debugging).
    """

    n = connections.shape[0]

    if ax is None:
        ax = plt.gca()

    ax.plot()

    ax.scatter(*points, c='k', alpha=.2)
    [ax.annotate(i, (points[0, i], points[1, i])) for i in range(n)]

    for i in range(n):
        for j in range(n):
            if connections[i, j]:
                ax.plot([points[0, i], points[0, j]], [points[1, i], points[1, j]], 'k', alpha=0.2)

def all_paths(p, r, n):
    
    rand_points = np.random.uniform(size=(2, n-2))
    edge_points = np.array([[0.0, 1.0],[0.0, 1.0]])
    points = np.concatenate((rand_points, edge_points), axis=1)

    connections = get_connections(points, pval=p, radius=r)

    paths = []

    paths.append(shortest_path(connections)[0][::-1])
    paths.append(longest_path(connections)[0][::-1])
    paths.append(shortest_path(connections.astype(bool))[0][::-1])
    paths.append(longest_path(connections.astype(bool))[0][::-1]) 

    forward_included = relevant_points(connections, sink=n-1)
    backward_included = relevant_points(connections.transpose(), sink=n-2)

    paths.append(greedy_path(connections, select_func=np.argmin, source=n-2, sink=n-1, included=forward_included)[0])
    paths.append(greedy_path(connections, select_func=np.argmax, source=n-2, sink=n-1, included=forward_included)[0])
    paths.append(greedy_path(connections.transpose(), select_func=np.argmin, source=n-1, sink=n-2, included=backward_included)[0][::-1])
    paths.append(greedy_path(connections.transpose(), select_func=np.argmax, source=n-1, sink=n-2, included=backward_included)[0][::-1])

    short_met, long_met, short_net, long_net, lMin_F, lMax_F, lMin_B, lMax_B = [[list(points[:, u]) for u in indexes] for indexes in paths]

    return(short_met, long_met, short_net, long_net, lMin_F, lMax_F, lMin_B, lMax_B)
           
def all_path_write_heads(fileName):
    headers = 'p - r - n - short_met - long_met - short_net - long_net - lMin_F - lMax_F - lMin_B - lMax_B'
    file1 = open("{}".format(fileName),"a") 
    file1.writelines(headers)
    file1.writelines('\n')
    file1.close()
           
def write_all_paths(fileName, p, r, n, all_paths):

            file1 = open("{}".format(fileName),"a") 
            file1.writelines('{} - {} - {} - {} - {} - {} - {} - {} - {} - {} - {}\n'.format(p, r, n, *all_paths))
            file1.close()

            return fileName

def all_paths_bulk(fileName, ps, ns, rs, repeats=1, verbose=True):
    
    all_path_write_heads(fileName)

    for p, n, r in zip(ps, ns, rs):

        if verbose:
            print(f'{p}-{n}-{r}', end=': ')
        
        for i in range(repeats):
        
            if verbose:
                print(i, end ='')
        
            paths = all_paths(p, r, n)

            write_all_paths(fileName, p, r, n, paths)
        
        if verbose:
            print()