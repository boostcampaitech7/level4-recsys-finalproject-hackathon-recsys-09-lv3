import torch

from model.layers import FactorizationMachine, MultiLayerPerceptron


class DeepFactorizationMachineModel(torch.nn.Module):

    def __init__(self, description, embed_dim, mlp_dims, dropout, item_id_name='item_id'):
        super().__init__()
        # assert len(description) == 12, 'unillegal format of {}'.format(description)
        orders = [name for name, _, _ in description] # description에 있는 feature 전부 쓰기
        
        # 아래는 feature 선택하기 위해 직접 설정하는 부분. feature selection하려면 아래 order 고쳐서 사용
        # orders = ['user_id', 'item_id', 'hours', 'date_release', 'positive_ratio', 'price_final', 'price_original', 'peak_ccu', 'required_age', 'price', 'metacritic_score', 'date', 'interaction', 'date', 'count']
        # orders = ['user_id', 'item_id', 'interaction']
        self.features = [name for name, _, type in description if (type != 'label') & (name in orders)]
        assert item_id_name in self.features, 'unkown item id name'
        self.description = {name: (size, type) for name, size, type in description if name in self.features}
        self.item_id_name = item_id_name
        self.build(embed_dim, mlp_dims, dropout)
    
    def build(self, embed_dim, mlp_dims, dropout):
        self.emb_layer = torch.nn.ModuleDict()
        self.ctn_emb_layer = torch.nn.ParameterDict()
        self.ctn_linear_layer = torch.nn.ModuleDict()
        self.embed_output_dim = 0
        for name, (size, type) in self.description.items():
            if type == 'spr':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                self.embed_output_dim += embed_dim
            elif type == 'ctn':
                self.ctn_emb_layer[name] = torch.nn.Parameter(torch.zeros([1, embed_dim], requires_grad=True))
                self.ctn_linear_layer[name] = torch.nn.Linear(1, 1, bias=False)
                self.embed_output_dim += embed_dim
            elif type == 'seq':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                self.embed_output_dim += embed_dim
            elif type == 'emb':
                self.emb_layer[name] = torch.nn.Linear(size, embed_dim, bias=False)
                self.embed_output_dim += embed_dim
            elif type == 'label':
                pass
            else:
                raise ValueError('unkown feature type: {}'.format(type))
        self.fm = FactorizationMachine()
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        return
    
    def init(self):
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def only_optimize_itemid(self):
        for name, param in self.named_parameters():
            if self.item_id_name not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        return
    
    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def forward(self, x_dict):
        item_id_emb = self.emb_layer[self.item_id_name](x_dict[self.item_id_name])
        loss = self.forward_with_item_id_emb(item_id_emb, x_dict)
        return loss

    def forward_with_item_id_emb(self, item_id_emb, x_dict):
        if item_id_emb.dim() == 2:
            item_id_emb = item_id_emb.unsqueeze(1)
        linears = []
        embs = []
        for name, (_, type) in self.description.items():
            if name == self.item_id_name or type == 'label':
                continue
            x = x_dict[name]
            if type == 'spr':
                embs.append(self.emb_layer[name](x))
            elif type == 'ctn':
                embs.append((self.ctn_emb_layer[name] * x).unsqueeze(1))
                linears.append(self.ctn_linear_layer[name](x))
            elif type == 'seq':
                embs.append(self.emb_layer[name](x).sum(dim=1, keepdims=True))
            elif type == 'emb':
                embs.append(self.emb_layer[name](x))
            else:
                raise ValueError('unkwon feature: {}'.format(name))
        emb = torch.concat([item_id_emb] + embs, dim=1)
        linear_part = torch.concat(linears, dim=1).sum(dim=1, keepdims=True) if linears else 0
        res = (linear_part + self.fm(emb) + self.mlp(emb.view(-1, self.embed_output_dim))).squeeze(dim=1)
        return torch.sigmoid(res)

