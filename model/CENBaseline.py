from tqdm import tqdm
import torch
import torch.nn as nn


class CEN(nn.Module):
    """
    Contextualized Regressor / Contextual Explanation Network
    y = w(c) * x
    no archetypes: w(c) = MLP(c)
    archetypes: w(c) = <softmax(MLP(c)), archetypes>
    """
    def __init__(self, x_dim, y_dim, c_dim, encoder_depth=3, encoder_width=25, num_archetypes=0):
        super(CEN, self).__init__()
        self.use_archetypes = num_archetypes > 0
        layers = [nn.Linear(c_dim, encoder_width), nn.ReLU()]
        for _ in range(encoder_depth - 2):
            layers += [nn.Linear(encoder_width, encoder_width), nn.ReLU()]
        if self.use_archetypes > 0:
            layers += [nn.Linear(encoder_width, num_archetypes), nn.Softmax(dim=1)]
            init_archetypes = torch.rand(num_archetypes, x_dim) * 1e-3
            self.archetypes = nn.parameter.Parameter(init_archetypes, requires_grad=True)
        else:
            layers += [nn.Linear(encoder_width, x_dim)]        
        self.mlp = nn.Sequential(*layers)

    def forward(self, c):
        if not self.use_archetypes:
            return self.mlp(c)
        z = self.mlp(c)
        batch_archetypes = self.archetypes.unsqueeze(0).repeat(z.shape[0], 1, 1)
        w = torch.bmm(z, batch_archetypes)
        return w
    
    def mse(self, x, y, c):
        w = self(c)
        return ((torch.bmm(w, x) - y) ** 2).mean()
    
    def fit(self, x, y, c, epochs, optimizer=torch.optim.Adam, lr=1e-3, silent=False):
        opt = optimizer(self.parameters(), lr=lr)
        progress_bar = tqdm(range(epochs), disable=silent)
        for epoch in progress_bar:
            loss = self.mse(x, y, c)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_desc = f'[Train MSE: {loss.item():.4f}] Epoch'
            progress_bar.set_description(train_desc)
            
    def predict_w(self, c):
        return self(c)

    def predict_y(self, x, c):
        return torch.bmm(self(c), x)


if __name__ == '__main__':
    # CEN demo
    n = 100
    c_dim = 10
    x_dim = c_dim
    y_dim = 1
    c = torch.rand((n, 1, c_dim)) * 2 - 1
    w = torch.clone(c) + 5
    x = torch.rand((n, x_dim, 1))
    y = torch.bmm(w, x)
    cen = CEN(x_dim, y_dim, c_dim, encoder_depth=3, num_archetypes=0)
    print(f"No Archetypes, Start MSE: {cen.mse(x, y, c)}")
    cen.fit(x, y, c, 1000)
    print(f"No Archetypes, Final MSE: {cen.mse(x, y, c)}")
    print()
    cen = CEN(x_dim, y_dim, c_dim, encoder_depth=3, num_archetypes=10)
    print(f"10 Archetypes, Start MSE: {cen.mse(x, y, c)}")
    cen.fit(x, y, c, 1000)
    print(f"10 Archetypes, Final MSE: {cen.mse(x, y, c)}")
    
