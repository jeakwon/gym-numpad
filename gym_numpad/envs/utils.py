import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy

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
        self._paths = []
        
    def __call__(self, edges, n):
        graph = Graph(edges, n)
        return self.findHamiltonianPaths(graph, n)
        
    def hamiltonianPaths(self, graph, v, visited, path, n):
 
        # if all the vertices are visited, then the Hamiltonian path exists
        if len(path) == n:
            # print the Hamiltonian path
            self._paths.append(deepcopy(path))
            return

        # Check if every edge starting from vertex `v` leads to a solution or not
        for w in graph.adjList[v]:

            # process only unvisited vertices as the Hamiltonian
            # path visit each vertex exactly once
            if not visited[w]:
                visited[w] = True
                path.append(w)

                # check if adding vertex `w` to the path leads to the solution or not
                self.hamiltonianPaths(graph, w, visited, path, n)

                # backtrack
                visited[w] = False
                path.pop()
                
    def findHamiltonianPaths(self, graph, n):

        # start with every node
        for start in range(n):

            # add starting node to the path
            path = [start]

            # mark the start node as visited
            visited = [False] * n
            visited[start] = True

            self.hamiltonianPaths(graph, start, visited, path, n)
        return self._paths
      
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
def create_2d_connected_sequences(m, n, diag_neighbors=True, seed=0):
    G = create_2d_graph(m, n, add_diag_edges=diag_neighbors)
    Gi = nx.convert_node_labels_to_integers(G, ordering = 'sorted', label_attribute = 'origin' )

    HPG = HamiltonianPathGenerator()
    paths = HPG(Gi.edges, len(Gi.nodes))
    
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
    return np.stack(arrs) 
