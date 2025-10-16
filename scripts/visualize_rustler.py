# %%
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from rt.data import RelationalDataset


# %%
def get_color(node_idx, offsets_desc):
    for i, (name, off) in enumerate(offsets_desc):
        if node_idx >= off:
            return name, i / (len(offsets_desc))


def viz(item, ax, dataset_name):

    is_task_nodes = item["is_task_nodes"][0].tolist()
    is_target_nodes = item["is_targets"][0].tolist()
    node_idxs = item["node_idxs"][0].tolist()
    f2p_nbr_idxs = item["f2p_nbr_idxs"][0].tolist()
    is_padding = item["is_padding"][0].tolist()

    file = Path(f"~/scratch/pre/{dataset_name}/table_info.json").expanduser()
    with open(file, "r") as f:
        table_json = json.load(f)

    offsets = {}
    for full_name, meta in table_json.items():
        base = full_name
        offsets[base] = int(meta["node_idx_offset"])

    offsets_desc = sorted(offsets.items(), key=lambda kv: kv[1], reverse=True)

    g = nx.DiGraph()
    target_nodes = []
    task_nodes = []
    db_nodes = []
    node_idx_set = set(node_idxs)

    subsequence_len = len(node_idxs)
    for i in range(len(node_idxs)):
        if is_padding[i]:
            continue

        node_idx = node_idxs[i]
        g.add_node(node_idx)

        if is_task_nodes[i]:
            table_name, color = get_color(node_idx, offsets_desc)
            task_nodes.append((node_idx, table_name, color))
            if is_target_nodes[i]:
                target_nodes.append((node_idx, table_name, color))
        else:
            table_name, color = get_color(node_idx, offsets_desc)
            db_nodes.append((node_idx, table_name, color))

        for nbr_idx in f2p_nbr_idxs[i]:
            if nbr_idx == -1 or nbr_idx not in node_idx_set:
                continue
            g.add_edge(node_idx, nbr_idx)

    task_nodes = list(set(task_nodes))
    db_nodes = list(set(db_nodes))
    target_nodes = list(set(target_nodes))

    pos = nx.nx_agraph.graphviz_layout(nx.Graph(g), prog="twopi", root=node_idxs[0])
    cmap = plt.get_cmap("tab20")

    if db_nodes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[node for node, _, _ in db_nodes],
            ax=ax,
            node_color=[color for _, _, color in db_nodes],
            cmap=cmap,
            vmin=0,
            vmax=1,
            node_size=100,
            node_shape="o",
            alpha=0.8,
        )

    if task_nodes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[node for node, _, _ in task_nodes],
            ax=ax,
            node_color=[color for _, _, color in task_nodes],
            cmap=cmap,
            vmin=0,
            vmax=1,
            node_size=120,
            node_shape="s",
            alpha=0.8,
        )

    if target_nodes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[node for node, _, _ in target_nodes],
            ax=ax,
            node_color="black",
            node_size=150,
            node_shape="^",
        )

    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        edge_color=(0.2, 0.2, 0.2, 0.6),
        arrows=True,
        arrowsize=10,
        arrowstyle="->",
    )

    legend_handles = []
    table_color_map = {}

    for _, table_name, color in task_nodes:
        table_color_map[(table_name, True)] = color
    for _, table_name, color in db_nodes:
        table_color_map[(table_name, False)] = color

    for (table_name, is_task), color in sorted(table_color_map.items()):
        shape = "□" if is_task else "○"
        legend_handles.append(
            Patch(facecolor=cmap(color), label=f"{shape} {table_name}")
        )

    legend_handles.extend(
        [
            Patch(facecolor="none", label=""),
            Patch(label="○ Dataset nodes"),
            Patch(label="□ Task nodes"),
            Patch(label="▲ Target nodes"),
            Patch(label="★ ICL seed nodes"),
        ]
    )

    ax.legend(
        handles=legend_handles,
        title="Node Types & Tables",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    num_nodes = len(g.nodes())
    num_edges = len(g.edges())
    num_targets = len(target_nodes)
    ax.set_title(f"Graph: {num_nodes} nodes, {num_edges} edges, {num_targets} targets")


# %%
dataset_name = "rel-stack"
# Load the dataset configuration
table_name = "user-badge"
target_column = "WillGetBadge"
dataset = RelationalDataset(
    tasks=[(dataset_name, table_name, target_column, "val", [])],
    batch_size=1,
    seq_len=1024,
    rank=0,
    world_size=1,
    max_bfs_width=256,
    embedding_model="all-MiniLM-L12-v2",
    d_text=384,
    seed=0,
)


# %%
# item = dataset[random.randint(0, len(dataset))]
item = dataset[40]
fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
ax.set_axis_off()
result = viz(item, ax, dataset_name)
fig.show()
