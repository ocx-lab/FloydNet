import torch

def transform_pcqm(graph):
    # Data(x=[36, 9], edge_index=[2, 72], edge_attr=[72, 3], edge_label_index=[2, 84], edge_label=[84])
    graph.x = torch.nn.functional.pad(graph.x, (0, 0, 1, 0))
    graph.edge_index = graph.edge_index + 1
    graph.edge_label_index = graph.edge_label_index + 1
    return graph