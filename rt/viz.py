
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def viz_full(item, db_text_list, task_text_list, ax):
    is_task_nodes = item["is_task_nodes"][0].tolist()
    node_idxs = item["node_idxs"][0].tolist()
    f2p_nbr_idxs = item["f2p_nbr_idxs"][0].tolist()
    table_name_idxs = item["table_name_idxs"][0].tolist()

    # TODO: fix coloring
    db_color_dict = {}
    task_color_dict = {}

    def get_db_color(id_):
        if id_ not in db_color_dict:
            db_color_dict[id_] = (len(db_color_dict) + 0.1) / 20
        return db_color_dict[id_]

    def get_task_color(id_):
        if id_ not in task_color_dict:
            task_color_dict[id_] = 0.5 + (len(task_color_dict) + 0.1) / 20
        return task_color_dict[id_]

    g = nx.DiGraph()
    node_colors = []
    table_colors = set()
    node_idx_set = set(node_idxs)
    for i in range(len(node_idxs)):
        if i != 0 and node_idxs[i] == node_idxs[i - 1]:
            continue

        node_idx = node_idxs[i]
        g.add_node(node_idx)

        table_name_idx = table_name_idxs[i]
        if is_task_nodes[i]:
            color = get_task_color(table_name_idx)
        else:
            color = get_db_color(table_name_idx)
        node_colors.append(color)
        is_task_node = is_task_nodes[i]
        if is_task_node:
            table_name = task_text_list[table_name_idx]
        else:
            table_name = db_text_list[table_name_idx]
        table_colors.add((table_name, color))

        for nbr_idx in f2p_nbr_idxs[i]:
            if nbr_idx == -1:
                continue
            if nbr_idx not in node_idx_set:
                continue
            g.add_edge(node_idx, nbr_idx)

    # pos = nx.bfs_layout(nx.Graph(g), start=node_idxs[0])
    pos = nx.nx_agraph.graphviz_layout(nx.Graph(g), prog="twopi", root=node_idxs[0])
    cmap = plt.get_cmap("prism")
    nx.draw_networkx(
        g,
        pos,
        ax=ax,
        with_labels=False,
        node_color=node_colors,
        cmap=cmap,
        vmin=0,
        vmax=1,
        node_size=100,
        node_shape="o",
        edge_color=(0.0, 0.0, 0.0, 0.5),
    )

    ax.legend(
        handles=[
            Patch(facecolor=cmap(color), label=table_name)
            for table_name, color in table_colors
        ],
        title="Tables",
    )
