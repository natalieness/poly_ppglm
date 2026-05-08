import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

def build_connectivity_graph(model) -> nx.DiGraph:
    """
    Neuron A -> B weighted directed graph, weights summed across all release sites.
    Ignores synapse co-occurrence / polyadic structure.
    """
    G = nx.DiGraph()
    G.add_nodes_from(model.neurons)
    for site in model.release_sites:
        for post in site.partners:
            if G.has_edge(site.pre, post):
                G[site.pre][post]["weight"] += site.weight
            else:
                G.add_edge(site.pre, post, weight=site.weight)
    return G


def plot_connectivity_graph(
    G: nx.DiGraph,
    inhibitory_neurons: Optional[set] = None,
    seed: int = 42,
) -> tuple:
    inhibitory_neurons = inhibitory_neurons or set()
    pos = nx.spring_layout(G, seed=seed)

    node_colors = [
        "#863ce7" if n in inhibitory_neurons else "#2ecc71"
        for n in G.nodes()
    ]

    exc_edges = [(u, v) for u, v, w in G.edges(data="weight") if w >= 0]
    inh_edges = [(u, v) for u, v, w in G.edges(data="weight") if w < 0]

    exc_widths = [abs(G[u][v]["weight"]) for u, v in exc_edges]
    inh_widths = [abs(G[u][v]["weight"]) for u, v in inh_edges]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edgelist=exc_edges, width=exc_widths,
        edge_color="#343e48", arrows=True, ax=ax,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=inh_edges, width=inh_widths,
        edge_color="#bf3ce7", arrows=True, ax=ax,
        connectionstyle="arc3,rad=0.1",
    )
    ax.set_title("Neuron connectivity")
    ax.axis("off")
    return fig, ax