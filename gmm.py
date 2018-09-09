import functools

import numpy as np
import torch
import torch.distributions as dist


class GMM(object):
    """GMM model.

    Args:
        means (list of list or torch.Tensor, optional): modes mean
        variances (ditto):
        priors (ditto):
        rng (numpy.RandomState, optional):
        seed (int, optional):
        dtype (torch.dtype, optional):
        mode_dtype (torch.dtype, optional):

    """

    def __init__(self, means=None, variances=None, priors=None, rng=None,
                 seed=None, dtype=torch.float, mode_dtype=torch.long):
        if means is None:
            means = list(map(
                lambda x: 10.0 * torch.tensor(x, dtype=dtype),
                [[0, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]
            ))

        means = list(map(
            lambda x: torch.tensor(x, dtype=dtype),
            means
        ))

        self.means = means
        self.num_components = len(self.means)
        self.dim = means[0].size(0)

        if variances is None:
            variances = [
                torch.eye(self.dim) for _ in range(self.num_components)
            ]
        self.variances = variances
        if priors is None:
            priors = [
                1.0 / self.num_components for _ in range(self.num_components)
            ]
        self.priors = priors

        assert len(means) == len(variances), \
            "Shape mismatch btwn means and variances | {}, {}".format(
                len(means), len(variances))
        assert len(variances) == len(priors), \
            "Shape mismatch btwn variances and priors | {}, {}".format(
                len(variances), len(priors))

        if rng is None:
            rng = np.random.RandomState(seed=seed)
        self.rng = rng
        self.dtype, self.mode_dtype = dtype, mode_dtype

    def _sample_prior(self, num_samples):
        return self.rng.choice(
            a=self.num_components, size=(num_samples,),
            replace=True, p=self.priors
        )

    def _sample_gaussian(self, mean, var):
        return dist.MultivariateNormal(mean, var).sample().view(1, -1)

    def sample(self, num_samples):
        fathers = self._sample_prior(num_samples)
        samples = torch.cat([
            self._sample_gaussian(self.means[father], self.variances[father])
            for father in fathers
        ], dim=0)
        return samples.type(self.dtype), \
            torch.from_numpy(fathers).type(self.mode_dtype).view(-1, 1)

    def _gaussian_pdf(self, x, mean, var):
        return dist.MultivariateNormal(mean, var).log_prob(x).exp()

    def pdf(self, x):
        """Calc pdf for a batch of samples or one sample."""
        pdfs = list(map(
            lambda m, v, p: p * self._gaussian_pdf(x, m, v), self.means, self.variances, self.priors
        ))

        return functools.reduce(lambda x, y: x + y, pdfs, 0.0)


class GMMData(object):

    def __init__(self, num_samples, means=None, variances=None,
                 priors=None, rng=None, seed=0, **kwargs):
        self.num_samples = num_samples
        if rng is None:
            seed = seed
            rng = np.random.RandomState(seed)

        _gm = GMM(means, variances, priors, rng, seed)
        self.means, self.variances, self.priors = _gm.means, _gm.variances, _gm.priors
        features, labels = _gm.sample(num_samples)
        densities = _gm.pdf(x=features)
        self.data = {
            'samples': features,
            'labels': labels,
            'densities': densities
        }

    def __len__(self):
        return self.num_samples

    @property
    def samples(self):
        return self.data['samples']

    @property
    def labels(self):
        return self.data['labels']

    @property
    def densities(self):
        return self.data['densities']

    def __getitem__(self, index):
        return self.samples[index], self.labels[index], self.densities[index]


if __name__ == '__main__':
    gmm = GMM([[0], [3], [5]])

    samples, modes = gmm.sample(10)
    print('num modes', gmm.num_components)
    print('modes', modes)
    print(samples.size(), modes.size())
    print('pdf', gmm.pdf(samples[0]))
    print('pdf', gmm.pdf(samples))

    gmm_dataset = GMMData(100)

    gmm_dataset.samples[:10]
