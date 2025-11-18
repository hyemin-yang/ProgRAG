from torch import autograd, nn
from collections.abc import Sequence
import torch
import copy
from torch_scatter import scatter_add
from torch_geometric.data import Data
from .nbflayer import *
from .nbftasks import *

class BaseNBFNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_relation,
        message_func="distmult",
        aggregate_func="pna",
        # aggregate_func="sum",
        short_cut=False,
        layer_norm=False,
        activation="relu",
        concat_hidden=False,
        num_mlp_layer=2,
        dependent=False,
        remove_one_hop=False,
        num_beam=10,
        path_topk=10,
        **kwargs,
    ):
        super().__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = (
            short_cut  # whether to use residual connections between GNN layers
        )
        self.concat_hidden = concat_hidden  # whether to compute final states as a function of all layer outputs or last
        self.remove_one_hop = remove_one_hop  # whether to dynamically remove one-hop edges from edge_index
        self.num_beam = num_beam
        self.path_topk = path_topk

        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.layer_norm = layer_norm
        self.activation = activation
        self.num_mlp_layers = num_mlp_layer

    def remove_easy_edges(self, data, h_index, t_index, r_index=None):
        # we remove training edges (we need to predict them at training time) from the edge index
        # think of it as a dynamic edge dropout
        h_index_ext = torch.cat([h_index, t_index], dim=-1)
        t_index_ext = torch.cat([t_index, h_index], dim=-1)
        r_index_ext = torch.cat([r_index, r_index + data.num_relations // 2], dim=-1)
        if self.remove_one_hop:
            # we remove all existing immediate edges between heads and tails in the batch
            edge_index = data.edge_index
            easy_edge = torch.stack([h_index_ext, t_index_ext]).flatten(1)
            index = edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(index, data.num_edges)
        else:
            # we remove existing immediate edges between heads and tails in the batch with the given relation
            edge_index = torch.cat([data.edge_index, data.edge_type.unsqueeze(0)])
            # note that here we add relation types r_index_ext to the matching query
            easy_edge = torch.stack([h_index_ext, t_index_ext, r_index_ext]).flatten(1)
            index = edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(index, data.num_edges)

        data = copy.copy(data)
        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]
        return data

    def negative_sample_to_tail(self, h_index, t_index, r_index, num_direct_rel):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        # new_h_index = torch.where(is_t_neg, h_index, t_index)
        # new_t_index = torch.where(is_t_neg, t_index, h_index)
        # new_r_index = torch.where(is_t_neg, r_index, r_index + num_direct_rel)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index)
        return new_h_index, new_t_index, new_r_index

    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)
        
        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(
                layer_input,
                query,
                boundary,
                data.edge_index,
                data.edge_type,
                size,
                edge_weight,
            )
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, batch):
        h_index, t_index, r_index = batch.unbind(-1)
        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            # data = self.remove_easy_edges(data, h_index, t_index, r_index, data.num_relations // 2)
            pass

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        
        output = self.bellmanford(
            data, h_index[:, 0], r_index[:, 0]
        )  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(
            1, index
        )  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)

    def visualize(self, data, batch):
        assert batch.shape == (1, 3)
        h_index, t_index, r_index = batch.unbind(-1)

        output = self.bellmanford(data, h_index, r_index, separate_grad=True)
        feature = output["node_feature"]
        edge_weights = output["edge_weights"]

        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index).squeeze(0)
        score = self.mlp(feature).squeeze(-1)

        edge_grads = autograd.grad(score, edge_weights)
        distances, back_edges = self.beam_search_distance(
            data, edge_grads, h_index, t_index, self.num_beam
        )
        paths, weights = self.topk_average_length(
            distances, back_edges, t_index, self.path_topk
        )

        return paths, weights

    @torch.no_grad()
    def beam_search_distance(self, data, edge_grads, h_index, t_index, num_beam=10):
        # beam search the top-k distance from h to t (and to every other node)
        num_nodes = data.num_nodes
        input = torch.full((num_nodes, num_beam), float("-inf"), device=h_index.device)
        input[h_index, 0] = 0
        edge_mask = data.edge_index[0, :] != t_index

        distances = []
        back_edges = []
        for edge_grad in edge_grads:
            # we don't allow any path goes out of t once it arrives at t
            node_in, node_out = data.edge_index[:, edge_mask]
            relation = data.edge_type[edge_mask]
            edge_grad = edge_grad[edge_mask]

            message = input[node_in] + edge_grad.unsqueeze(-1)  # (num_edges, num_beam)
            # (num_edges, num_beam, 3)
            msg_source = (
                torch.stack([node_in, node_out, relation], dim=-1)
                .unsqueeze(1)
                .expand(-1, num_beam, -1)
            )

            # (num_edges, num_beam)
            is_duplicate = torch.isclose(
                message.unsqueeze(-1), message.unsqueeze(-2)
            ) & (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            # pick the first occurrence as the ranking in the previous node's beam
            # this makes deduplication easier later
            # and store it in msg_source
            is_duplicate = is_duplicate.float() - torch.arange(
                num_beam, dtype=torch.float, device=message.device
            ) / (num_beam + 1)
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat(
                [msg_source, prev_rank], dim=-1
            )  # (num_edges, num_beam, 4)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            # sort messages w.r.t. node_out
            message = message[order].flatten()  # (num_edges * num_beam)
            msg_source = msg_source[order].flatten(0, -2)  # (num_edges * num_beam, 4)
            size = node_out.bincount(minlength=num_nodes)
            msg2out = size_to_index(size[node_out_set] * num_beam)
            # deduplicate messages that are from the same source and the same beam
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat(
                [torch.zeros(1, dtype=torch.bool, device=message.device), is_duplicate]
            )
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = msg2out.bincount(minlength=len(node_out_set))

            if not torch.isinf(message).all():
                # take the topk messages from the neighborhood
                # distance: (len(node_out_set) * num_beam)
                distance, rel_index = scatter_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                # store msg_source for backtracking
                back_edge = msg_source[abs_index]  # (len(node_out_set) * num_beam, 4)
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                # scatter distance / back_edge back to all nodes
                distance = scatter_add(
                    distance, node_out_set, dim=0, dim_size=num_nodes
                )  # (num_nodes, num_beam)
                back_edge = scatter_add(
                    back_edge, node_out_set, dim=0, dim_size=num_nodes
                )  # (num_nodes, num_beam, 4)
            else:
                distance = torch.full(
                    (num_nodes, num_beam), float("-inf"), device=message.device
                )
                back_edge = torch.zeros(
                    num_nodes, num_beam, 4, dtype=torch.long, device=message.device
                )

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        # backtrack distances and back_edges to generate the paths
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, prev_rank) in zip(
                distance[:k].tolist(), back_edge[:k].tolist()
            ):
                if d == float("-inf"):
                    break
                path = [(h, t, r)]
                for j in range(i - 1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(
                *sorted(zip(average_lengths, paths), reverse=True)[:k]
            )

        return paths, average_lengths

class EntityNBFNet(BaseNBFNet):
    """Neural Bellman-Ford Network for Entity Prediction.

    This class extends BaseNBFNet to perform entity prediction in knowledge graphs using a neural
    version of the Bellman-Ford algorithm. It learns entity representations through message passing
    over the graph structure.

    Args:
        input_dim (int): Dimension of input node/relation features
        hidden_dims (list): List of hidden dimensions for each layer
        num_relation (int, optional): Number of relation types. Defaults to 1 (dummy value)
        **kwargs: Additional arguments passed to BaseNBFNet

    Attributes:
        layers (nn.ModuleList): List of GeneralizedRelationalConv layers
        mlp (nn.Sequential): Multi-layer perceptron for final prediction
        query (torch.Tensor): Relation type embeddings used as queries

    Methods:
        bellmanford(data, h_index, r_index, separate_grad=False):
            Performs neural Bellman-Ford message passing iterations.

            Args:
                data: Graph data object containing edge information
                h_index (torch.Tensor): Indices of head entities
                r_index (torch.Tensor): Indices of relations
                separate_grad (bool): Whether to use separate gradients for visualization

            Returns:
                dict: Contains node features and edge weights after message passing

        forward(data, relation_representations, batch):
            Forward pass for entity prediction.

            Args:
                data: Graph data object
                relation_representations (torch.Tensor): Embeddings of relations
                batch: Batch of (head, tail, relation) triples

            Returns:
                torch.Tensor: Prediction scores for tail entities
    """
    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):
        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                GeneralizedRelationalConv(
                    self.dims[i],
                    self.dims[i + 1],
                    num_relation,
                    self.dims[0],
                    self.message_func,
                    self.aggregate_func,
                    self.layer_norm,
                    self.activation,
                    dependent=False,
                    project_relations=True,
                )
            )

        feature_dim = (
            sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]
        ) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)
        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        # query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)
        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(
                layer_input,
                query,
                boundary,
                data.edge_index,
                data.edge_type,
                size,
                edge_weight,
            )
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(
            -1, data.num_nodes, -1
        )  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch):
        h_index, t_index, r_index = batch.unbind(-1)
        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            # data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index, num_direct_rel=data.num_relations // 2
        )
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(
            1, index
        )  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)

class QueryGNN(nn.Module):
    """A neural network module for query embedding in graph neural networks.

    This class implements a query embedding model that combines relation embeddings with an entity-based graph neural network
    for knowledge graph completion tasks.

    Args:
        entity_model (EntityNBFNet): The entity-based neural network model for reasoning on graph structure.
        rel_emb_dim (int): Dimension of the relation embeddings.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        rel_emb_dim (int): Dimension of relation embeddings.
        entity_model (EntityNBFNet): The entity model instance.
        rel_mlp (nn.Linear): Linear transformation layer for relation embeddings.

    Methods:
        forward(data: Data, batch: torch.Tensor) -> torch.Tensor:
            Forward pass of the query GNN model.

            Args:
                data (Data): Graph data object containing the knowledge graph structure and features.
                batch (torch.Tensor): Batch of triples with shape (batch_size, 1+num_negatives, 3),
                                    where each triple contains (head, tail, relation) indices.

            Returns:
                torch.Tensor: Scoring tensor for the input triples.
    """

    def __init__(
        self, entity_model: EntityNBFNet, rel_emb_dim: int, *args, **kwargs
    ):
        """Initialize the model.

        Args:
            entity_model (EntityNBFNet): The entity model component
            rel_emb_dim (int): Dimension of relation embeddings
            *args (Any): Variable length argument list
            **kwargs (Any): Arbitrary keyword arguments

        """

        super().__init__()
        self.rel_emb_dim = rel_emb_dim
        self.entity_model = entity_model
        self.rel_mlp = nn.Linear(rel_emb_dim, self.entity_model.dims[0])

    def forward(self, data: Data, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            data (Data): Graph data object containing entity embeddings and graph structure.
            batch (torch.Tensor): Batch of triple indices with shape (batch_size, 1+num_negatives, 3),
                                where each triple contains (head_idx, tail_idx, relation_idx).

        Returns:
            torch.Tensor: Scores for the triples in the batch.

        Notes:
            - Relations are assumed to be the same across all positive and negative triples
            - Easy edges are removed before processing to encourage learning of non-trivial paths
            - The batch tensor contains both positive and negative samples where the first sample
              is positive and the rest are negative samples
        """
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        batch_size = len(batch)
        relation_representations = (
            self.rel_mlp(data.rel_emb).unsqueeze(0).expand(batch_size, -1, -1)
        )
        h_index, t_index, r_index = batch.unbind(-1)
        # to make NBFNet iteration learn non-trivial paths
        # data = self.entity_model.remove_easy_edges(data, h_index, t_index, r_index)

        score = self.entity_model(data, relation_representations, batch)

        return score

class QueryNBFNet(EntityNBFNet):
    def bellmanford(self, data, node_features, query, separate_grad=False):
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=query.device)

        hiddens = []
        edge_weights = []
        layer_input = node_features

        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            hidden = layer(
                layer_input,
                query,
                node_features,
                data.edge_index,
                data.edge_type,
                size,
                edge_weight,
            )
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        node_query = query.unsqueeze(1).expand(
            -1, data.num_nodes, -1
        )  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, node_features, relation_representations, query):
        for layer in self.layers:
            layer.relation = relation_representations
        output = self.bellmanford(
            data, node_features, query
        )  # (num_nodes, batch_size, feature_dim）
        score = self.mlp(output["node_feature"]).squeeze(-1)  # (bs, num_nodes)
        return score, output["node_feature"]

class GNNRetriever(QueryGNN):
    def __init__(self, entity_model: QueryNBFNet, rel_emb_dim: int) -> None:

        super().__init__(entity_model, rel_emb_dim)
        self.question_mlp = nn.Linear(self.rel_emb_dim, self.entity_model.dims[0])

    def forward(self, graph, batch, entities_weight,):
        question_emb = batch["question_embeddings"].unsqueeze(0)
        question_entities_mask = batch["question_entities_masks"].unsqueeze(0)
        question_embedding = self.question_mlp(question_emb)  # shape: (bs, emb_dim)
        batch_size = question_embedding.size(0)
        relation_representations = (self.rel_mlp(graph.rel_emb).unsqueeze(0).expand(batch_size, -1, -1))

        # initialize the input with the fuzzy set and question embs
        if entities_weight is not None:
            question_entities_mask = question_entities_mask * entities_weight.unsqueeze(
                0
            )
        input = torch.einsum(
            "bn, bd -> bnd", question_entities_mask, question_embedding
        )

        # GNN model: run the entity-level reasoner to get a scalar distribution over nodes
        output, emb = self.entity_model(
            graph, input, relation_representations, question_embedding
        )

        return output, emb
    
def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

def size_to_index(size):
    range = torch.arange(len(size), device=size.device)
    index2sample = range.repeat_interleave(size)
    return index2sample

def multi_slice_mask(starts, ends, length):
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    mask = scatter_add(values, slices, dim=0, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask

def scatter_extend(data, size, input, input_size):
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = torch.zeros(
        new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device
    )
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = input
    return new_data, new_size

def scatter_topk(input, size, k, largest=True):
    index2graph = size_to_index(size)
    index2graph = index2graph.view([-1] + [1] * (input.ndim - 1))

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    safe_input = input.clamp(2 * min - max, 2 * max - min)
    offset = (max - min) * 4
    if largest:
        offset = -offset
    input_ext = safe_input + offset * index2graph
    index_ext = input_ext.argsort(dim=0, descending=largest)
    num_actual = size.clamp(max=k)
    num_padding = k - num_actual
    starts = size.cumsum(0) - size
    ends = starts + num_actual
    mask = multi_slice_mask(starts, ends, len(index_ext)).nonzero().flatten()

    if (num_padding > 0).any():
        # special case: size < k, pad with the last valid index
        padding = ends - 1
        padding2graph = size_to_index(num_padding)
        mask = scatter_extend(mask, num_actual, padding[padding2graph], num_padding)[0]

    index = index_ext[mask]  # (N * k, ...)
    value = input.gather(0, index)
    if isinstance(k, torch.Tensor) and k.shape == size.shape:
        value = value.view(-1, *input.shape[1:])
        index = index.view(-1, *input.shape[1:])
        index = index - (size.cumsum(0) - size).repeat_interleave(k).view(
            [-1] + [1] * (index.ndim - 1)
        )
    else:
        value = value.view(-1, k, *input.shape[1:])
        index = index.view(-1, k, *input.shape[1:])
        index = index - (size.cumsum(0) - size).view([-1] + [1] * (index.ndim - 1))

    return value, index
