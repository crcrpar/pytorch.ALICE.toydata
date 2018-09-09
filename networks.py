import torch
import torch.nn as nn


def _linear_relu(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())


# Model=========================================================================
class _Model(nn.Module):

    def __init__(self, num_layers, in_features, noise_features,
                 hidden_features, out_features, **kwargs):
        super(_Model, self).__init__()

        self.noise_features = noise_features
        dim_list = [in_features + noise_features] + [hidden_features] * num_layers
        layers = [_linear_relu(in_, out_) for in_, out_ in zip(dim_list, dim_list[1:])]
        layers.append(nn.Linear(hidden_features, out_features))
        self.model = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        device = kwargs.pop('device', torch.device('cuda'))
        x = torch.cat([x, torch.randn(x.size(0), self.noise_features).type(x.dtype).to(device)], dim=1)
        return self.model(x)


class Generator(_Model):

    """Returns `p_x`."""

    def __init__(self, num_layers, in_features, noise_features,
                 hidden_features, out_features, **kwargs):
        super(Generator, self).__init__(
            num_layers, in_features, noise_features,
            hidden_features, out_features, **kwargs
        )


class Inference(_Model):

    """Returns `q_z`."""

    def __init__(self, num_layers, in_features, noise_features,
                 hidden_features, out_features, **kwargs):
        super(Inference, self).__init__(
            num_layers, in_features, noise_features,
            hidden_features, out_features, **kwargs
        )


class Discriminator(_Model):

    def __init__(self, num_layers, in_features, noise_features,
                 hidden_features, out_features, **kwargs):
        super(Discriminator, self).__init__(
            num_layers, in_features, noise_features,
            hidden_features, out_features, **kwargs
        )

    def forward(self, x1, x2, **kwargs):
        x = torch.cat([x1, x2], dim=1)
        return self.model(x).squeeze(dim=1)


class DiscriminatorXX(Discriminator):

    """Approximate x log data density."""

    def __init__(self, num_layers, in_features, noise_features,
                 hidden_features, out_features, **kwargs):
        super(DiscriminatorXX, self).__init__(
            num_layers, in_features, noise_features,
            hidden_features, out_features, **kwargs
        )


class DiscriminatorZZ(Discriminator):

    """Approximate x log data density."""

    def __init__(self, num_layers, in_features, noise_features,
                 hidden_features, out_features, **kwargs):
        super(DiscriminatorZZ, self).__init__(
            num_layers, in_features, noise_features,
            hidden_features, out_features, **kwargs
        )
# Model=========================================================================
