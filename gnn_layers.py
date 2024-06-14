import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
    
class GAT(nn.Module):

    def __init__(self, in_features, out_features, dropout, negative_slope, num_heads=1, bias=True, self_loop_type=2, average=False, normalize=False):
        super(GAT, self).__init__()
        """
        Initialization method for the graph-attentional layer

        Arguments:
            in_features (int): number of features in each input node
            out_features (int): number of features in each output node
            dropout (int/float): dropout probability for the coefficients
            negative_slope (int/float): control the angle of the negative slope in leakyrelu
            number_heads (int): number of heads of attention
            bias (bool): if adding bias to the output
            self_loop_type (int): 0 -- force no self-loop; 1 -- force self-loop; other values (2)-- keep the input adjacency matrix unchanged
            average (bool): if averaging all attention heads
            normalize (bool): if normalizing the coefficients after zeroing out weights using the communication graph
        """

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_heads = num_heads
        self.self_loop_type = self_loop_type
        self.average = average
        self.normalize = normalize

        self.W = nn.Parameter(torch.zeros(size=(in_features, num_heads * out_features)))
        self.a_i = nn.Parameter(torch.zeros(size=(num_heads, out_features, 1)))
        self.a_j = nn.Parameter(torch.zeros(size=(num_heads, out_features, 1)))
        if bias:
            if average:
                self.bias = nn.Parameter(torch.DoubleTensor(out_features))
            else:
                self.bias = nn.Parameter(torch.DoubleTensor(num_heads * out_features))
        else:
            self.register_parameter('bias', None)
        self.leakyrelu = nn.LeakyReLU(self.negative_slope)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialization for the parameters of the graph-attentional layer
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W.data, gain=gain)
        nn.init.xavier_normal_(self.a_i.data, gain=gain)
        nn.init.xavier_normal_(self.a_j.data, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, input, adj):
        """

        Arguments:
            input (tensor): input of the graph attention layer [N * in_features, N: number of agents]
            adj (tensor): the learned communication graph (adjancy matrix) by the sub-scheduler [N * N]

        Return:
            the output of the graph attention layer
        """

        # perform linear transformation on the input, and generate multiple heads
        # self.W: [in_features * (num_heads*out_features)]
        # h (tensor): the matrix after performing the linear transformation [N * num_heads * out_features]
        h = torch.mm(input, self.W).view(-1, self.num_heads, self.out_features)
        N = h.size()[0]
    
        # forbid the self-loop
        if self.self_loop_type == 0:
            adj = adj * (torch.ones(N, N) - torch.eye(N, N))
        # self-loop can happen
        elif self.self_loop_type == 1:
            adj = torch.eye(N, N, device=adj.device) + adj * (torch.ones(N, N, device=adj.device) - torch.eye(N, N, device=adj.device))   
        # keep unchanged
        else:
            pass
        
        e = []

        # compute the unnormalized coefficients
        # a_i, a_j (tensors): weight vectors to compute the unnormalized coefficients [num_heads * out_features * 1]
        for head in range(self.num_heads):
            # coeff_i, coeff_j (tensors): intermediate matrices to calculate unnormalized coefficients [N * 1]
            coeff_i = torch.mm(h[:, head, :], self.a_i[head, :, :])
            coeff_j = torch.mm(h[:, head, :], self.a_j[head, :, :])
            # coeff (tensor): the matrix of unnormalized coefficients for each head [N * N * 1]
            coeff = coeff_i.expand(N, N) + coeff_j.transpose(0, 1).expand(N, N)
            coeff = coeff.unsqueeze(-1)
            
            e.append(coeff)
            
        # e (tensor): the matrix of unnormalized coefficients for all heads [N * N * num_heads]
        # sometimes the unnormalized coefficients can be large, so regularization might be used 
        # to limit the large unnormalized coefficient values (TODO)
        e = self.leakyrelu(torch.cat(e, dim=-1)) 
            
        # adj: [N * N * num_heads]
        adj = adj.unsqueeze(-1).expand(N, N, self.num_heads)
        # attention (tensor): the matrix of coefficients used for the message aggregation [N * N * num_heads]
        attention = e * adj
        attention = F.softmax(attention, dim=1)
        # the weights from agents that should not communicate (send messages) will be 0, the gradients from 
        # the communication graph will be preserved in this way
        attention = attention * adj   
        # normalize: make the some of weights from all agents be 1
        if self.normalize:
            if self.self_loop_type != 1:
                attention += 1e-15
            attention = attention / attention.sum(dim=1).unsqueeze(dim=1).expand(N, N, self.num_heads)
            attention = attention * adj
        # dropout on the coefficients  
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # output (tensor): the matrix of output of the gat layer [N * (num_heads*out_features)]
        output = []
        for head in range(self.num_heads):
            h_prime = torch.matmul(attention[:, :, head], h[:, head, :])
            output.append(h_prime)
        if self.average:
            output = torch.mean(torch.stack(output, dim=-1), dim=-1)
        else:
            output = torch.cat(output, dim=-1)
        
        if self.bias is not None:
            output += self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(in_features={}, out_features={})'.format(self.in_features, self.out_features)
    
class HyperGraphConv(nn.Module):
    def __init__(self, n_edges):
        super(HyperGraphConv, self).__init__()
        # print(n_edges)
        self.W_line = nn.Parameter(torch.ones(n_edges).cuda())
        self.W = None

    def forward(self, node_features, hyper_graph):
        self.W = torch.diag_embed(self.W_line)
        B_inv = torch.sum(hyper_graph.detach(), dim=-2)
        B_inv = torch.diag_embed(B_inv)
        softmax_w = torch.abs(self.W).detach()
        D_inv = torch.matmul(hyper_graph.detach(), softmax_w).sum(dim=-1)
        D_inv = torch.diag_embed(D_inv)
        D_inv = D_inv **(-0.5)
        B_inv = B_inv **(-1)
        D_inv[D_inv == float('inf')] = 0
        D_inv[D_inv == float('nan')] = 0
        B_inv[B_inv == float('inf')] = 0
        B_inv[B_inv == float('nan')] = 0
        A = torch.bmm(D_inv, hyper_graph)
        A = torch.matmul(A, torch.abs(self.W))
        A = torch.bmm(A, B_inv)
        A = torch.bmm(A, hyper_graph.transpose(-2, -1))
        A = torch.bmm(A, D_inv)
        X = torch.bmm(A, node_features)
        return X
    
    
class HyperGraph(nn.Module):
    def __init__(self, args):
        super(HyperGraph, self).__init__()
        self.args = args
        self.n_agents = args.nagents
        # self.input_dim = args.hid_size * 3 # (encoded_obs, hidden_state, cell_state)
        # self.input_dim = args.hid_size  
        
        self.encoder = nn.Linear(args.obs_size, args.hid_size) # encode obs  
        
        self.wq = nn.Linear(args.obs_size, args.qk_hid_size)
        self.wk = nn.Linear(args.obs_size, args.qk_hid_size)
        
        
        self.node2edge_mlp = nn.Sequential(
            nn.Linear(in_features=args.hid_size, out_features=args.hid_size * 2),
            nn.ReLU(),
            nn.Linear(in_features=args.hid_size * 2, out_features=args.hid_size),
            nn.ReLU(),
        )
        self.edge2node_mlp = nn.Sequential(
            nn.Linear(in_features=args.hid_size * 2, out_features=args.hid_size * 2),
            nn.ReLU(),
            nn.Linear(in_features=args.hid_size * 2, out_features=args.hid_size),
            nn.ReLU(),
        )
        
        self.attention_mlp = nn.Sequential(
            nn.Linear(in_features=args.hid_size * 2, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
            nn.ReLU(),
        ) 
    
    def Hypergraph_generator(self, hidden_state, relate_matrix, scale_factor=2):
        batch = hidden_state.shape[0]
        agent_num = hidden_state.shape[1]
        if scale_factor == agent_num:
            H_matrix = torch.ones(batch,1,agent_num).type_as(hidden_state)
            return H_matrix
        group_size = scale_factor
        if group_size < 1:
            group_size = 1
        # feat_corr: (batch,actor_number,actor_number) the affinity matrix
        # choose the top k correlation node for each node on the 2nd dimension
        _,indice = torch.topk(relate_matrix,dim=2,k=group_size,largest=True) 
        H_matrix = torch.zeros(batch,agent_num,agent_num).type_as(hidden_state)
        H_matrix = H_matrix.scatter(2,indice,1)

        return H_matrix   # H[i][j] == 1 means that i-th agent should correlated with j-th hyperedge


    def node2edge(self, x, H):
        x = self.node2edge_mlp(x)
        edge_init = torch.matmul(H,x) # edge_init: (batch,edge_number,h_dim) # aggregate embeddings of related nodes for each node
        node_num = x.shape[1]
        edge_num = edge_init.shape[1]
        x_rep = (x[:,:,None,:].transpose(2,1)).repeat(1,edge_num,1,1) # (batch,actor_number,actor_number,h_dim)
        edge_rep = edge_init[:,:,None,:].repeat(1,1,node_num,1) # (batch,actor_number,actor_number,h_dim)
        node_edge_cat = torch.cat((x_rep,edge_rep),dim=-1)
        attention_weight = self.attention_mlp(node_edge_cat)[:,:,:,0] # (batch,actor_number,actor_number)
        H_weight = attention_weight * H
        H_weight = F.softmax(H_weight,dim=2)
        H_weight = H_weight * H
        edges = torch.matmul(H_weight,x) # (batch,edge_number,h_dim)
        return edges
    
    
    def edge2node(self, edge_feat, node_feat, H):
        node_feature = torch.cat((torch.matmul(H.permute(0,2,1), edge_feat), node_feat),dim=-1)
        node_feature = self.edge2node_mlp(node_feature / node_feature.shape[1])
        return node_feature
    
    def edgefeat_generate(self, hidden_state): # from observations generate hypergraph
        
        # TODO: use respective wq and wk to compute attention, similar to ga-comm
        agent_features = F.relu(self.encoder(hidden_state)) # from observations generate features for each agent
        
        
        q = self.wq(hidden_state)
        # k = self.wk(hidden_state)
        # size of 1 * N x N
        similar_matrix = torch.matmul(q, q.permute(0,2,1)) / np.sqrt(self.args.qk_hid_size)        
        
        # similar_matrix = torch.matmul(query, query.permute(0,2,1)) # (agent_num,agent_num) similarity matrix, relate_matrix[i][j] means the similarity between i-th agent and j-th agent
        # H[i][j] == 1 means that i-th agent is correlated with j-th agent. Each agent maintains a hyperedge
        H = self.Hypergraph_generator(agent_features, similar_matrix, scale_factor=agent_features.shape[1] // 4) # e.g. agent_num == 40, choose top10 agents
        edges = self.node2edge(agent_features, H)
        return edges, agent_features, H
    
    def forward(self, hidden_state):
        # TODO: obs can be replaced by hidden_state generated by LSTM
        # hidden_state = hidden_state.unsqueeze(0)
        edge_features, agent_features, H = self.edgefeat_generate(hidden_state) # from observations generate hyperedge features
        agent_features = self.edge2node(edge_features, agent_features, H)
        # entropy_loss = self.calculate_entropy_loss(hidden_state, hidden_state)
        return agent_features, H # H[i][j] == 1: i-th agent is in j-th edge
    
    def calculate_entropy_loss(self, q, k):
        query = self.wq(q.detach())
        key = self.wk(k.detach())
        alpha = torch.matmul(query, key.permute(0,2,1)) / np.sqrt(self.args.qk_hid_size)  
        alpha = torch.clamp(alpha, min=1e-4)
        entropy_loss = - (alpha * torch.log2(alpha)).sum(-1).mean()
        return entropy_loss * 0.001
    
    
    