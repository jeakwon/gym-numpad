import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
from functools import reduce

# edited from https://www.techiedelight.com/print-all-hamiltonian-path-present-in-a-graph/
# A class to represent a graph object
class Graph:
 
    # Constructor
    def __init__(self, edges, n):
 
        # A list of lists to represent an adjacency list
        self.adjList = [[] for _ in range(n)]
 
        # add edges to the undirected graph
        for (src, dest) in edges:
            self.adjList[src].append(dest)
            self.adjList[dest].append(src)

class HamiltonianPathGenerator:
    def __init__(self):
        self._stop_flag = False
        
    def __call__(self, edges, n, num_paths_per_nodes=None):
        """
        :params edges: edge list
        :params n: number of nodes
        :params num_paths_per_nodes: how many paths to find per node
        """
        self._paths = [[] for _ in range(n)]
        graph = Graph(edges, n)
        return self.findHamiltonianPaths(graph, n, num_paths_per_nodes)
        
    def hamiltonianPaths(self, graph, v, visited, path, n, N=None):
        if self._stop_flag:
            return
        
        # if all the vertices are visited, then the Hamiltonian path exists
        if len(path) == n:
            # print the Hamiltonian path
            init_node = path[0]
            self._paths[init_node].append(deepcopy(path))
            if N and len(self._paths[init_node])>=N:
#                 print(self._paths[0], len(self._paths[0]),N)
                self._stop_flag = True
            return

        # Check if every edge starting from vertex `v` leads to a solution or not
        for w in graph.adjList[v]:

            # process only unvisited vertices as the Hamiltonian
            # path visit each vertex exactly once
            if not visited[w]:
                visited[w] = True
                path.append(w)

                # check if adding vertex `w` to the path leads to the solution or not
                self.hamiltonianPaths(graph, w, visited, path, n, N)

                # backtrack
                visited[w] = False
                path.pop()
    
    def findHamiltonianPaths(self, graph, n, num_paths_per_nodes=None):
        # start with every node
        for start in range(n):
            self._stop_flag = False # continue finding if changed node
            
            # add starting node to the path
            path = [start]

            # mark the start node as visited
            visited = [False] * n
            visited[start] = True

            self.hamiltonianPaths(graph, start, visited, path, n, N=num_paths_per_nodes)
        return reduce(lambda x, y: x+y, self._paths) # concat list of per node path list into single list
      
def create_2d_graph(m, n, add_diag_edges=True, draw=False):
    G = nx.grid_2d_graph(m, n)

    if add_diag_edges:
        # https://stackoverflow.com/questions/55772715/how-to-create-8-cell-adjacency-map-for-a-diagonal-enabled-a-algorithm-with-the
        G.add_edges_from([
            ((x, y), (x+1, y+1))
            for x in range(m-1)
            for y in range(n-1)
        ] + [
            ((x+1, y), (x, y+1))
            for x in range(m-1)
            for y in range(n-1)
        ], weight=1.414)

    if draw:
        pos = nx.spring_layout(G, iterations=100, seed=39775)
        nx.draw(G, pos, node_size=0, with_labels=False)
    return G
   
def create_2d_connected_sequences(m, n, diag_neighbors=True, seed=0, num_paths=None):
    G = create_2d_graph(m, n, add_diag_edges=diag_neighbors)
    Gi = nx.convert_node_labels_to_integers(G, ordering = 'sorted', label_attribute = 'origin' )
    num_nodes = len(Gi.nodes)
    
    num_paths_per_nodes = num_paths//num_nodes+1
    
    HPG = HamiltonianPathGenerator()
    paths = HPG(Gi.edges, num_nodes, num_paths_per_nodes)
    
    node_pos = list(G.nodes)
    arrs = []
    for path in paths:
        arr = np.zeros(shape=(m, n), dtype=np.int32)
        for seq, k in enumerate(path):
            i, j = list(G.nodes)[k]
            arr[i, j] = seq+1
        arrs.append(arr)

    rng = np.random.default_rng(seed)
    rng.shuffle(arrs)
    return np.stack(arrs)[:num_paths]
