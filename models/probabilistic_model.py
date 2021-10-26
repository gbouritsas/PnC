import torch

class ProbabilisticModel(torch.nn.Module):

    def __init__(self,
                 max_dict_size,
                 b_max,
                 b_distribution='uniform',
                 delta_distribution='uniform',
                 atom_distribution='uniform',
                 cut_size_distribution='uniform',
                 b_min=1,
                 c_max=None):
        super(ProbabilisticModel, self).__init__()
        self.max_dict_size = max_dict_size
        # x_a = binary variable indicating the existence of an atom in the dictionary or not
        self.x_a_logits = torch.nn.Parameter(torch.zeros((max_dict_size,)))
        # Regarding the initialisation: we initialise the membership variables close to 1
        # since we observed it facilitates optimisation
        # (e.g., it allows the NeuralPart to initially focus on good partitions rather than frequent subgraphs)
        torch.nn.init.constant_(self.x_a_logits, 4)

        self.b_max = b_max
        self.b_min = b_min
        self.b_distribution = b_distribution
        if b_distribution=='learnable':
            # p(b) = probability of number of blocks in the partition
            self.b_logits = torch.nn.Parameter(torch.zeros((b_max - b_min + 1,)))
            torch.nn.init.zeros_(self.b_logits)
        self.delta_distribution = delta_distribution
        if delta_distribution=='learnable':
            # delta = probability of emission of a non-dictionary subgraph
            self.subgraph_in_dict_logit = torch.nn.Parameter(torch.zeros((1,)))
            torch.nn.init.zeros_(self.subgraph_in_dict_logit)
        self.atom_distribution = atom_distribution
        if atom_distribution == 'learnable':
            # p(a) = probability of emission of atom a
            self.atom_logits = torch.nn.Parameter(torch.zeros((max_dict_size,)))
            torch.nn.init.zeros_(self.atom_logits)
        self.cut_size_distribution = cut_size_distribution
        if cut_size_distribution=='learnable':
            self.cut_size_logits = torch.nn.Parameter(torch.zeros((c_max,)))
            torch.nn.init.zeros_(self.cut_size_logits)
        self.softmax = torch.nn.Softmax(dim=0)
        self.sigmoid = torch.nn.Sigmoid()

    def membership_vars(self, current_universe_size=None):
        if current_universe_size is None:
            current_universe_size = len(self.x_a_logits)
        x_a = torch.zeros_like(self.x_a_logits)
        x_a[0:current_universe_size] = self.sigmoid(self.x_a_logits[0:current_universe_size])
        return x_a

    def b_probs(self):
        return self.softmax(self.b_logits)

    def delta_prob(self):
        return 1 - self.sigmoid(self.subgraph_in_dict_logit)

    def atom_probs(self, x_a):
        p_a = torch.zeros_like(self.atom_logits)
        mask_probs = x_a != 0
        p_a[mask_probs] = self.softmax(x_a[mask_probs].log() + self.atom_logits[mask_probs])
        return p_a

    def cut_size_probs(self):
        return self.softmax(self.cut_size_logits)

    def prune_universe(self, train):
        # atoms in dict
        # x_a =  dictionary_probs_model.atoms_bernoulli(len(agent.env.dictionary))
        x_a = self.membership_vars()
        if train:
            x_a_hard = (x_a > 0.5).float()
            x_a =  (x_a_hard - x_a).detach() + x_a
        else:
            x_a = (x_a > 0.5).float()
        if self.b_distribution == 'uniform':
            b_probs = 1 / (self.b_max - self.b_min + 1) *\
                      torch.ones((self.b_max - self.b_min + 1,), device=x_a.device)  # add +1 here if empty graphs are allowed
        elif self.b_distribution == 'learnable':
            b_probs = self.b_probs()
        else:
            b_probs = torch.ones((self.b_max - self.b_min + 1,), device=x_a.device)

        if self.atom_distribution == 'uniform':
            atom_probs = x_a / x_a.sum() if x_a.sum() != 0 else torch.zeros_like(x_a)
        else:
            atom_probs = self.atom_probs(x_a)

        if self.delta_distribution == 'uniform':
            delta_prob = 1 / 2 * torch.ones((1,), device=x_a.device)
        elif self.delta_distribution == 'learnable':
            delta_prob = self.delta_prob()
        else:
            delta_prob = torch.ones((1,), device=x_a.device)

        if self.cut_size_distribution == 'uniform':
            cut_size_probs=None
        else:
            cut_size_probs = self.cut_size_probs()
        return x_a, b_probs, delta_prob, atom_probs, cut_size_probs

    def count_params(self, x_a):
        num_params = 0
        if self.b_distribution == 'learnable':
            num_params += self.b_logits.numel()
        if self.delta_distribution == 'learnable':
            num_params += self.subgraph_in_dict_logit.numel()
        if self.atom_distribution == 'learnable':
            num_params += x_a.sum().item()
        if self.cut_size_distribution == 'learnable':
            num_params += self.cut_size_logits.numel()
        return num_params


