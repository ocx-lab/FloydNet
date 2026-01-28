from collections import defaultdict

import torch

def _eval_mrr(y_pred_pos, y_pred_neg):
    """ Compute Hits@k and Mean Reciprocal Rank (MRR).

    Implementation from OGB:
    https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py

    Args:
        y_pred_neg: array with shape (batch size, num_entities_neg).
        y_pred_pos: array with shape (batch size, )
    """

    y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
    argsort = torch.argsort(y_pred, dim=1, descending=True)
    ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
    ranking_list = ranking_list[:, 1] + 1
    # average within graph
    hits1 = (ranking_list <= 1).to(torch.float).mean().item()
    hits3 = (ranking_list <= 3).to(torch.float).mean().item()
    hits10 = (ranking_list <= 10).to(torch.float).mean().item()
    mrr = (1. / ranking_list.to(torch.float)).mean().item()

    # print(f"hits@1  {hits1:.5f}")
    # print(f"hits@3  {hits3:.5f}")
    # print(f"hits@10 {hits10:.5f}")
    # print(f"mrr     {mrr:.5f}")
    return hits1, hits3, hits10, mrr


class EdgeMRR:
    def __init__(self):
        self.states = defaultdict(lambda: [])

    def clean(self):
        self.states = defaultdict(lambda: [])

    def add_batch(self, pred, graph_batch):
        pred = pred[-1]
        for b in range(graph_batch.single_mask.shape[0]):
            indices = torch.where(graph_batch.single_mask[b])[0]
            self.states["preds"].append(pred[b][indices][:, indices])
            self.states["trues"].append(graph_batch.adj_label[b][indices][:, indices])

    def compute(self):
        # pred: list of [n, n]
        # true: list of [n, n]
        pred_list = self.states["preds"]
        true_list = self.states["trues"]
        batch_stats = [[], [], [], []]
        for pred, true in zip(pred_list, true_list):
            n = pred.shape[0]
            pos_edge_index = torch.where(true == 1)
            pred_pos = pred[pos_edge_index]
            num_pos_edges = pos_edge_index[0].shape[0]
            if num_pos_edges == 0:
                continue

            neg_mask = torch.ones([num_pos_edges, n], dtype=torch.bool)
            neg_mask[torch.arange(num_pos_edges), pos_edge_index[1]] = False
            pred_neg = pred[pos_edge_index[0]][neg_mask].view(num_pos_edges, -1)

            mrr_list = _eval_mrr(pred_pos, pred_neg)
            for i, v in enumerate(mrr_list):
                batch_stats[i].append(v)
        # average among all graphs
        res = []
        for i in range(4):
            v = torch.tensor(batch_stats[i])
            v = torch.nan_to_num(v, nan=0).sum().item()
            res.append(v)

        return {
            'hits@1': res[0],
            'hits@3': res[1],
            'hits@10': res[2],
            'mrr': res[3],
            "sample_count": len(pred_list),
        }