import torch
from torch import nn
import torch_geometric.nn as gnn

from cross_attention_graph import CrossAttentionGraphModule
from emulator_bench.modeling import build_graph_plan, cross_attention, lcm_pool

NODE_COUNTS = [1, 2, 3, 5, 8, 7, 1, 4]


def test_lcm_pool_matches_per_graph_aggregation():
    torch.manual_seed(0)
    aggregation = gnn.LCMAggregation(8, 16)
    x = torch.randn(sum(NODE_COUNTS), 8)
    pooled = lcm_pool(aggregation, x, build_graph_plan(NODE_COUNTS))
    expected = torch.cat([aggregation(part) for part in x.split(NODE_COUNTS)])
    assert pooled.shape == (len(NODE_COUNTS), 16)
    torch.testing.assert_close(pooled, expected, atol=1e-6, rtol=1e-5)


def test_cross_attention_matches_padded_module():
    torch.manual_seed(0)
    module = CrossAttentionGraphModule(num_heads=4, node_feature_size=16, latent_size=24, dropout=0.0).eval()
    lengths = torch.tensor([6, 3, 9, 1, 4, 2, 9, 5])
    max_len = int(lengths.max())
    nodes = torch.randn(sum(NODE_COUNTS), 16)
    states = [torch.randn(int(n), 24) for n in lengths]
    padded = nn.utils.rnn.pad_sequence(states, batch_first=True)
    padding_mask = torch.arange(max_len)[None] >= lengths[:, None]
    token_index = torch.cat([torch.arange(int(n)) + row * max_len for row, n in enumerate(lengths)])

    packed = cross_attention(module, nodes, build_graph_plan(NODE_COUNTS), torch.cat(states), token_index, padding_mask)
    batch = torch.repeat_interleave(torch.arange(len(NODE_COUNTS)), torch.tensor(NODE_COUNTS))
    expected, _ = module(nodes, batch, padded, padding_mask, return_attention_weights=True)
    torch.testing.assert_close(packed, expected, atol=1e-5, rtol=1e-4)
