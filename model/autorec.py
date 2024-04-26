import torch
import torch.nn as nn


class AutoRec(nn.Module):
    def __init__(self, num_input, hidden_units):
        super().__init__()
        self.encoder = nn.Linear(num_input, hidden_units)
        self.decoder = nn.Linear(hidden_units, num_input)

    def forward(self, x):
        hidden = torch.sigmoid(self.encoder(x))
        output = torch.sigmoid(self.decoder(hidden))
        return output

    def recommendation(self, rating_matrix, k):
        prediction = self.forward(rating_matrix)
        values, indices = torch.topk(prediction, k, dim=1)
        return indices.cpu().numpy()

    def i_recommendation(self, rating_matrix, k):
        prediction = self.forward(rating_matrix)
        values, indices = torch.topk(prediction, k, dim=0)
        return indices.cpu().numpy()
