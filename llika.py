import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt


def networkBuildKnn(
    X_net,
    Y_net,
    knn=5,
    e_percentile=None,
    class_connected=False,
    metric="euclidean",
    neighbors=True,
    colors=[],
):
    g = nx.Graph()
    g.graph["knn"] = knn
    g.graph["e_percentile"] = e_percentile
    g.graph["class_connected"] = class_connected
    g.graph["metric"] = metric
    g.graph["neighbors"] = neighbors

    lnNet = len(X_net)
    g.graph["class_names"] = list(set(Y_net))
    g.graph["colors"] = colors
    class_nodes = [[] for i in g.graph["class_names"]]

    for index, instance in enumerate(X_net):
        label = Y_net[index]
        index_label = g.graph["class_names"].index(label)
        class_nodes[index_label].append(str(index))
        g.add_node(str(index), value=instance, type_node="net", label=label)
    g.graph["class_nodes"] = class_nodes

    values = X_net
    if values.ndim == 1:
        values = np.reshape(values, (-1, 1))

    nbrs = NearestNeighbors(n_neighbors=knn + 1, metric=metric)
    nbrs.fit(values)

    distances, indices = nbrs.kneighbors(values)
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    for indice_node, indices_node in enumerate(indices):
        for tmpi, indice in enumerate(indices_node):
            if (
                g.nodes()[str(indice)]["label"] == g.nodes()[
                    str(indice_node)]["label"]
                or class_connected
            ):
                g.add_edge(
                    str(indice), str(indice_node), weight=distances[indice_node][tmpi]
                )

    if not e_percentile == None:
        eRadius = np.quantile(distances, e_percentile)
        nbrs.set_params(radius=eRadius)
        distances, indices = nbrs.radius_neighbors(values)

        for indice_node, indices_node in enumerate(indices):
            for tmpi, indice in enumerate(indices_node):
                if not str(indice) == str(indice_node):
                    if (
                        g.nodes()[str(indice)]["label"]
                        == g.nodes()[str(indice_node)]["label"]
                        or class_connected
                    ):
                        g.add_edge(
                            str(indice),
                            str(indice_node),
                            weight=distances[indice_node][tmpi],
                        )
    g.graph["index"] = lnNet
    if neighbors:
        g.graph["nbrs"] = nbrs

    return g


def insertNode(g, instance, label="?", colors=[]):
    node_index = g.graph["index"]
    nbrs = g.graph["nbrs"]
    g.graph["index"] += 1
    colors = g.graph["colors"]
    class_names = g.graph["class_names"]
    class_nodes = g.graph["class_nodes"]

    g.add_node(str(node_index), value=instance, type_node="opt", label=label)
    if label == "?":
        color = "#000000"
    else:
        color = colors[class_names.index(label)]

    # if instance.ndim == 1:
    #     instance = np.reshape(instance, (-1, 1))

    distances, indices = nbrs.kneighbors([instance])
    indices = indices[:, :-1]
    distances = distances[:, :-1]

    for indice_node, indices_node in enumerate(indices):
        for tmpi, indice in enumerate(indices_node):
            if(not str(indice) == str(indice_node)):
                if(label == "?" or label == g.nodes()[str(indice)]["label"]):
                    g.add_edge(
                        str(indice),
                        str(node_index),
                        weight=distances[indice_node][tmpi],
                        color=color,
                    )

    tmpRadius = g.graph["e_percentile"]
    if not tmpRadius == None:
        distances, indices = nbrs.radius_neighbors(instance)
        for indice_node, indices_node in enumerate(indices):
            for tmpi, indice in enumerate(indices_node):
                if (not str(indice) == str(indice_node)):
                    if(label == "?" or label == g.nodes()[str(indice)]["label"]):
                        g.add_edge(
                            str(indice),
                            str(node_index),
                            weight=distances[indice_node][tmpi],
                            color=color,
                        )
    if label == "?":
        for index, e in enumerate(class_names):
            index_label = g.graph["class_names"].index(e)
            class_nodes[index_label].append(str(node_index))
    else:
        index_label = g.graph["class_names"].index(label)
        class_nodes[index_label].append(str(node_index))


def drawGraph(g, title="", sizeGraph=(10, 10), labels=False):
    plt.figure("Graph", figsize=sizeGraph)
    if "mod" in g.graph:
        mod = g.graph["mod"]
        plt.title(title + " Q:" + str(round(mod, 4)))
    else:
        plt.title(title)

    pos = nx.spring_layout(g, k=0.90, iterations=200, seed=42)
    # pos = nx.spring_layout(g, k=0.3,iterations=100,seed=42)
    # pos = nx.kamada_kawai_layout(g)
    color_group = g.graph["colors"]
    class_names = g.graph["class_names"]
    node_color = []
    edge_color = []
    # print("CLASES:", classes)
    for node, label in g.nodes(data="label"):
        if g.nodes[node]["type_node"] == "net":
            node_color.append(color_group[class_names.index(label)])
        if g.nodes[node]["type_node"] == "opt":
            node_color.append("#000000")
    for node_a, node_b, color in g.edges.data("color", default="#9db4c0"):
        edge_color.append("#a6a6a8")
    nx.draw(
        g,
        pos,
        node_color=node_color,
        edge_color=edge_color,
        width=1.3,
        node_size=50,
        with_labels=labels,
    )
    plt.show()


class Llika():
    network_type = "knn"
    knn = 5
    e_percentile = None
    class_connected = False
    metric = "euclidean"
    neighbors = True
    global_colors = [
        "#0b5d1e",
        "#004E98",
        "#E8871E",
        "#806443",
        "#725AC1",
        "#5386E4",
        "#ff6700",
        "#49111c",
        "#EF271B",
        "#937666",
        "#B0E298",
        "#1d7874",
        "#da627d",
        "#587B7F",
    ]
    # def setSettings(self, network_type="knn_e", knn=3, e_percentile=None, class_connected=False, similarity="euclidean", colors=None ):

    def setSettings(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def drawGraph(self, g, title, size_graph=(10, 10), labels=False):
        drawGraph(g, title, size_graph, labels)

    def buildNetwork(self, X_net, Y_net):
        if(self.network_type == "knn"):
            return networkBuildKnn(X_net, Y_net, self.knn, None, self.class_connected, self.metric, colors=self.global_colors)
        if(self.network_type == "e"):
            return networkBuildKnn(X_net, Y_net, 0, None, self.class_connected, self.metric, colors=self.global_colors)
        if(self.network_type == "knn_e"):
            return networkBuildKnn(X_net, Y_net, self.knn, self.e_percentile, self.class_connected, self.metric, colors=self.global_colors)

    def insertNode(self, g, instance, label="?"):
        insertNode(g, instance, label, self.global_colors)
    def removeLastNode(self,g):
        index = g.graph["index"]-1
        g.remove_node(index)
        g.graph["index"]=index