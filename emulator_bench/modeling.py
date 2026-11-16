from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch_geometric.nn as gnn

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cross_attention_graph import CrossAttentionGraphModule
from emulator_bench.common import HEAD_ORDER
from emulator_bench.feature_pipeline import ESM_TOKEN_SCALE


@dataclass
class GraphPlan:
    """Index tensors for one chunk: where each node sits in the padded (rows, max_nodes) grid, and the
    pairing of nodes at each level of the LCM pooling tree."""

    dense_index: torch.Tensor
    max_nodes: int
    rows: int
    levels: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    def apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "GraphPlan":
        return GraphPlan(fn(self.dense_index), self.max_nodes, self.rows, [tuple(fn(t) for t in level) for level in self.levels])


def build_graph_plan(node_counts) -> GraphPlan:
    """Plan from the node count of every graph in a chunk (nodes are stored graph by graph)."""
    counts = np.asarray(node_counts, dtype=np.int64)
    rows = len(counts)
    max_nodes = int(counts.max())
    graph_ids = np.repeat(np.arange(rows), counts)
    positions = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    levels = []
    lens = counts.copy()
    while lens.max() > 1:
        starts = np.cumsum(lens) - lens
        pairs = np.where(lens > 1, lens // 2, 0)
        odd = lens % 2 == 1
        pair_graph = np.repeat(np.arange(rows), pairs)
        pair_pos = np.arange(pairs.sum()) - np.repeat(np.cumsum(pairs) - pairs, pairs)
        left = starts[pair_graph] + 2 * pair_pos
        new_lens = pairs + odd
        new_starts = np.cumsum(new_lens) - new_lens
        odd_graph = np.nonzero(odd)[0]
        leftover = starts[odd_graph] + lens[odd_graph] - 1
        dest = np.concatenate([new_starts[pair_graph] + pair_pos, new_starts[odd_graph] + pairs[odd_graph]])
        levels.append(tuple(torch.from_numpy(a) for a in (left, leftover, dest)))
        lens = new_lens
    return GraphPlan(torch.from_numpy(graph_ids * max_nodes + positions), max_nodes, rows, levels)


def lcm_pool(aggregation: gnn.LCMAggregation, x: torch.Tensor, plan: GraphPlan) -> torch.Tensor:
    """LCMAggregation of every graph, one GRU call per tree level for all graphs together."""
    x = aggregation.lin(x).relu()
    for left, leftover, dest in plan.levels:
        a, b = x[left], x[left + 1]
        merged = aggregation.gru_cell(torch.cat([a, b]), torch.cat([b, a])).view(2, left.shape[0], -1).mean(dim=0)
        rest = x[leftover].to(merged.dtype)
        x = torch.zeros(dest.shape[0], merged.shape[1], dtype=merged.dtype, device=x.device).index_copy(0, dest, torch.cat([merged, rest]))
    return x


def cross_attention(
    module: CrossAttentionGraphModule,
    nodes: torch.Tensor,
    plan: GraphPlan,
    states: torch.Tensor,
    token_index: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Graph nodes attend to the unpadded residue states (total residues, features)."""
    rows, max_nodes = plan.rows, plan.max_nodes
    dense = torch.zeros(rows * max_nodes, nodes.shape[1], dtype=torch.float32, device=nodes.device)
    dense = dense.index_copy(0, plan.dense_index, nodes.float()).view(rows, max_nodes, -1)
    if not torch.is_autocast_enabled():
        states = states.float()
    kv = []
    for projection in (module.k_dense, module.v_dense):
        projected = projection(states)
        padded = torch.zeros(rows * padding_mask.shape[1], projected.shape[1], dtype=projected.dtype, device=projected.device)
        kv.append(padded.index_copy(0, token_index, projected).view(rows, padding_mask.shape[1], -1))
    attended, _ = module.attention(module.q_dense(dense), kv[0], kv[1], key_padding_mask=padding_mask, need_weights=True, average_attn_weights=False)
    x = module.ln1(attended + dense)
    x = module.leaky_relu(module.dense1(x)) + x
    x = module.ln2(x)
    return x.reshape(rows * max_nodes, -1)[plan.dense_index]


class BindRegressionModel(nn.Module):
    """BIND with the released layers. The forward runs the same maths as the released code with batched kernels:
    level-wise LCM pooling, dense cross-attention built by one scatter, and K/V projections on unpadded residues."""

    def __init__(
        self,
        esm_table,
        head_order: List[str] = None,
        cross_attention_dropout: float = 0.1,
        leaky_relu_negative_slope: float = 0.01,
    ):
        super().__init__()
        # ESM-2 token embeddings: layer 0 of the protein states is rebuilt on the GPU from the token ids.
        self.register_buffer("esm_table", torch.as_tensor(esm_table, dtype=torch.float32), persistent=False)
        self.head_order = list(head_order or HEAD_ORDER)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

        self.x_embed = nn.Linear(15, 16)
        self.x_embed_ln = gnn.LayerNorm(16)

        self.e_embed = nn.Linear(5, 2)
        self.e_embed_ln = nn.LayerNorm(2)

        self.conv1 = gnn.GATv2Conv(16, 64, edge_dim=2, negative_slope=0.01)
        self.ln1 = gnn.LayerNorm(64)
        self.crossattention1 = CrossAttentionGraphModule(num_heads=16, node_feature_size=64, latent_size=1280, dropout=cross_attention_dropout)

        self.conv2 = gnn.GATv2Conv(64, 64, edge_dim=2, negative_slope=0.01)
        self.ln2 = gnn.LayerNorm(64)
        self.crossattention2 = CrossAttentionGraphModule(num_heads=16, node_feature_size=64, latent_size=1280, dropout=cross_attention_dropout)

        self.conv3 = gnn.GATv2Conv(64, 64, edge_dim=2, negative_slope=0.01)
        self.ln3 = gnn.LayerNorm(64)
        self.crossattention3 = CrossAttentionGraphModule(num_heads=16, node_feature_size=64, latent_size=1280, dropout=cross_attention_dropout)

        self.conv4 = gnn.GATv2Conv(64, 64, edge_dim=2, negative_slope=0.01)
        self.ln4 = gnn.LayerNorm(64)
        self.crossattention4 = CrossAttentionGraphModule(num_heads=16, node_feature_size=64, latent_size=1280, dropout=cross_attention_dropout)

        self.conv5 = gnn.GATv2Conv(64, 64, edge_dim=2, negative_slope=0.01)
        self.ln5 = gnn.LayerNorm(64)

        self.pool = gnn.LCMAggregation(64, 1024)

        self.dense1 = nn.Linear(1024, 1024)
        self.lnf1 = nn.LayerNorm(1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.lnf2 = nn.LayerNorm(1024)

        self.heads = nn.ModuleDict({head_name: nn.Linear(1024, 1) for head_name in self.head_order})

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """batch is a BindBatch on the model device. Output: one (rows, 1) tensor per head."""
        padding_mask = torch.arange(batch.max_len, device=batch.lengths.device)[None, :] >= batch.lengths[:, None]
        states = [self.esm_table[batch.tokens.long()] * ESM_TOKEN_SCALE, *batch.flat.unbind(0)]

        graph = batch.graphs.sort()
        x, edge_index, edge_features = graph.x.float(), graph.edge_index, graph.edge_features.float()
        x = self.x_embed_ln(self.leaky_relu(self.x_embed(x)), graph.batch)
        edge_features = self.e_embed_ln(self.leaky_relu(self.e_embed(edge_features)))

        blocks = [
            (self.conv1, self.ln1, self.crossattention1),
            (self.conv2, self.ln2, self.crossattention2),
            (self.conv3, self.ln3, self.crossattention3),
            (self.conv4, self.ln4, self.crossattention4),
        ]
        for (conv, norm, attention), layer_states in zip(blocks, states):
            x = norm(self.leaky_relu(conv(x, edge_index, edge_features)), graph.batch)
            x = cross_attention(attention, x, batch.plan, layer_states, batch.token_index, padding_mask)
        x = self.ln5(self.leaky_relu(self.conv5(x, edge_index, edge_features)), graph.batch)

        x = lcm_pool(self.pool, x, batch.plan)
        x = self.lnf1(self.leaky_relu(self.dense1(x)))
        x = self.lnf2(self.leaky_relu(self.dense2(x)))
        return {head_name: head_layer(x) for head_name, head_layer in self.heads.items()}


def build_model(
    esm_table,
    head_order: List[str] = None,
    cross_attention_dropout: float = 0.1,
    leaky_relu_negative_slope: float = 0.01,
):
    return BindRegressionModel(
        esm_table,
        head_order=head_order or HEAD_ORDER,
        cross_attention_dropout=cross_attention_dropout,
        leaky_relu_negative_slope=leaky_relu_negative_slope,
    )
