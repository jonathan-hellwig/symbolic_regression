import torch
import sympy as sp

class SymbolicLayer:
    def __init__(self) -> None:
        pass

    def forward(self, x):
        x = torch.cat((x[:, 0, None], torch.sin(x[:, 1, None]), torch.cos(
            x[:, 2, None]), torch.sigmoid(x[:, 3, None]), x[:, 4, None] * x[:, 5, None]), 1)
        return x

    def sp_forward(self, x):
        return sp.Matrix([[x[0], sp.sin(x[1]), sp.cos(x[2]), 1 / (1 + sp.exp(-x[3])), x[4] * x[5]]])
        
    def parameters(self):
        return []


class LinearLayer:
    def __init__(self, in_dim, out_dim) -> None:
        self.W = torch.randn((in_dim, out_dim))
        self.b = torch.randn((1, out_dim))

    def forward(self, x):
        x = x @ self.W + self.b
        return x

    def sp_forward(self, x):
        return x * sp.Matrix(self.W) + sp.Matrix(self.b)

    def parameters(self):
        return [self.W, self.b]
