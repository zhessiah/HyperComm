import torch
import torch.nn.functional as F
from torch import nn
from gnn_layers import GAT, HyperGraph

class HyperComm(nn.Module):
    def __init__(self, args):
        super(HyperComm, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
        
        dropout = 0
        negative_slope = 0.2

        # process and distribute messages to agents
        self.distributer = GAT(args.hid_size, args.gat_hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads, self_loop_type=args.self_loop_type1, average=False, normalize=args.first_gat_normalize)
        # initialize the gat encoder for the Scheduler
        if args.use_gat_encoder:
            self.comm_encoder = GAT(args.hid_size, args.gat_encoder_out_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.ge_num_heads, self_loop_type=1, average=True, normalize=args.gat_encoder_normalize)
        
        self.obs_encoder = nn.Linear(args.obs_size, args.hid_size)

        self.init_hidden(args.batch_size)
        self.lstm_cell= nn.LSTMCell(args.hid_size, args.hid_size).double()

        # initialize mlp layers for the sub-schedulers
        if not args.first_graph_complete:
            if args.use_gat_encoder:
                self.gumbel_softmax_mlp = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.gumbel_softmax_mlp = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))
                
        if args.message_encoder:
            self.message_encoder = nn.Linear(args.hid_size, args.hid_size)
        if args.message_decoder:
            self.message_decoder = nn.Linear(args.hid_size, args.hid_size)

        # initialize weights as 0
        if args.comm_init == 'zeros':
            if args.message_encoder:
                self.message_encoder.weight.data.zero_()
            if args.message_decoder:
                self.message_decoder.weight.data.zero_()
            if not args.first_graph_complete:
                self.gumbel_softmax_mlp.apply(self.init_linear)

                   
        # initialize the action head (in practice, one action head is used)
        self.action_heads = nn.ModuleList([nn.Linear(2*args.hid_size, o)
                                        for o in args.naction_heads])
        # initialize the value head
        self.value_head = nn.Linear(2 * self.hid_size, 1)
        
        self.Hypergraph_Encoder = HyperGraph(args)
        

        self.to(self.device)


    def forward(self, x, info={}):
        """


        Arguments:
            x (list): a list for the input of the communication protocol [observations, (previous hidden states, previous cell states)]
            observations (tensor): the observations for all agents [1 (batch_size) * n * obs_size]
            previous hidden/cell states (tensor): the hidden/cell states from the previous time steps [n * hid_size]

        Returns:
            action_out (list): a list of tensors of size [1 (batch_size) * n * num_actions] that represent output policy distributions
            value_head (tensor): estimated values [n * 1]
            next hidden/cell states (tensor): next hidden/cell states [n * hid_size]
        """

        # n: number of agents

        obs, extras = x

        # encoded_obs: [1 (batch_size) * n * hid_size]
        encoded_obs = self.obs_encoder(obs)
        hidden_state, cell_state = extras

        batch_size = encoded_obs.size()[0]
        n = self.nagents

        _, agent_mask = self.get_agent_mask(batch_size, info)

        # if self.args.comm_mask_zero == True, block the communiction (can also comment out the protocol to make training faster)
        if self.args.comm_mask_zero:
            agent_mask *= torch.zeros(n, 1)

        hidden_state, cell_state = self.lstm_cell(encoded_obs.squeeze(), (hidden_state, cell_state))

        
        # hypergraph generation
        # TODO: obs can be replaced by hidden_state. However, according to the properties of MDP, obs is more reasonable.
        comm, H_matrix = self.Hypergraph_Encoder(obs) # comm: [batch(1), agent_num, hid_size] H_matrix: [batch(1), agent_num, agent_num]
        comm = comm.squeeze() # if not squeeze, bug will occurs
        H_matrix = H_matrix.squeeze() # assign H to the communication graph
        Hyper_incidence = torch.matmul(H_matrix, H_matrix.t())
        Hyper_incidence = torch.where(Hyper_incidence > 0, torch.ones_like(Hyper_incidence), torch.zeros_like(Hyper_incidence))
        
            
        # mask communcation from dead agents
        comm = comm * agent_mask
        # generate communication graph
        if self.args.use_gat_encoder:
            adj_complete = self.get_complete_graph(agent_mask)
            encoded_state = self.comm_encoder(comm, adj_complete)
            adj = self.topo_generator(self.gumbel_softmax_mlp, encoded_state, agent_mask, self.args.directed)
        else: # default
            adj = self.topo_generator(self.gumbel_softmax_mlp, comm, agent_mask, self.args.directed)


        comm = F.elu(self.distributer(comm, adj))
        
        
        
        # mask communication to dead agents
        comm = comm * agent_mask
        
        if self.args.message_decoder:
            comm = self.message_decoder(comm)

        value_head = self.value_head(torch.cat((hidden_state, comm), dim=-1))
        h = hidden_state.view(batch_size, n, self.hid_size)
        c = comm.view(batch_size, n, self.hid_size)

        action_out = [F.log_softmax(action_head(torch.cat((h, c), dim=-1)), dim=-1) for action_head in self.action_heads]

        return action_out, value_head, (hidden_state.clone(), cell_state.clone())

    def get_agent_mask(self, batch_size, info):
        """
        Function to generate agent mask to mask out inactive agents (only effective in Traffic Junction)

        Returns:
            num_agents_alive (int): number of active agents
            agent_mask (tensor): [n, 1]
        """

        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()
        
        agent_mask = agent_mask.to(self.device)

        return num_agents_alive, agent_mask

    def init_linear(self, m):
        """
        Function to initialize the parameters in nn.Linear as o 
        """
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)
        
    def init_hidden(self, batch_size):
        """
        Function to initialize the hidden states and cell states
        """
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True, device=self.device, dtype=torch.float64),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True, device=self.device, dtype=torch.float64)))
    
    
    def topo_generator(self, gumbel_softmax_mlp, hidden_state, agent_mask, directed=True):
        """
        generate the communication topology

        hidden_state [n * hid_size]
        agent_mask (tensor): [n * 1]
        directed (bool)

        Return:
            adj : comm topo [n * n]  
        """

        # hidden_state: [n * hid_size]
        n = self.args.nagents
        hid_size = hidden_state.size(-1)
        # hard_attn_input: [n * n * (2*hid_size)]
        # hidden_state.repeat(1, n): for each agent, repeat n times. hidden_state.repeat(n, 1) for each graph, repeat n times. Thus, the concatenation is (ei,ej) for all j, for all i
        hard_attn_input = torch.cat([hidden_state.repeat(1, n).view(n * n, -1), hidden_state.repeat(n, 1)], dim=1).view(n, -1, 2 * hid_size)
        # hard_attn_output: [n * n * 2]
        if directed:
            hard_attn_output = F.gumbel_softmax(gumbel_softmax_mlp(hard_attn_input), hard=True)
        else:
            hard_attn_output = F.gumbel_softmax(0.5*gumbel_softmax_mlp(hard_attn_input)+0.5*gumbel_softmax_mlp(hard_attn_input.permute(1,0,2)), hard=True)
        # chunk at 3rd dimension(dim 2), start at 1, length 1
        # hard_attn_output: [n * n * 2] -> [n * n * 1]
        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1) 
        # agent_mask and agent_mask_transpose: [n * n]
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        # adj: [n * n]
        adj = hard_attn_output.squeeze() * agent_mask * agent_mask_transpose
        
        return adj
    
    def get_complete_graph(self, agent_mask):
        """
        Function to generate a complete graph, and mask it with agent_mask
        """
        n = self.args.nagents
        adj = torch.ones(n, n, device=self.device)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = adj * agent_mask * agent_mask_transpose
        
        return adj    

    def calculate_entropy_loss(self, alpha):
        alpha = torch.clamp(alpha, min=1e-4)
        entropy_loss = - (alpha * torch.log2(alpha)).sum(-1).mean()
        return entropy_loss * self.args.entropy_loss_weight
