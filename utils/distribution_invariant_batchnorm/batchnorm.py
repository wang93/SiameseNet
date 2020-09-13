import torch
from torch.nn.modules.batchnorm import _BatchNorm as origin_BN
from warnings import warn


class _BatchNorm(origin_BN):
    @staticmethod
    def expand(stat, target_size):
        if len(target_size) == 4:
            stat = stat.unsqueeze(1).unsqueeze(2).expand(target_size[1:])
        elif len(target_size) == 2:
            pass
        else:
            raise NotImplementedError

        return stat

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input: torch.Tensor, indices):
        self._check_input_dim(input)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            means = []
            vars = []
            if input.dim() == 4:
                reduced_dim = (0, 2, 3)
            elif input.dim() == 2:
                reduced_dim = (0, )
            else:
                raise NotImplementedError

            data = input.detach()
            for group in indices:
                if len(group) == 0:
                    warn('There is no sample of one class in current batch, which may .')
                    continue
                samples = data[group]
                mean = torch.mean(samples, dim=reduced_dim, keepdim=False)
                var = torch.var(samples, dim=reduced_dim, keepdim=False)

                means.append(mean)
                vars.append(var)

            di_mean = torch.tensor(means).detach().mean(dim=0, keepdim=False)
            di_var = torch.tensor(vars).detach().mean(dim=0, keepdim=False)

            if self.track_running_stats:
                self.running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * di_mean
                self.running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * di_var
            else:
                self.running_mean = di_mean
                self.running_var = di_var

        sz = input.size()
        y = (input - self.expand(self.running_mean, sz)) \
            / self.expand(self.eps +torch.sqrt(self.running_var), sz)

        if self.affine:
            z = y * self.expand(self.weight, sz) + self.expand(self.bias, sz)
        else:
            z = y

        return z


class BatchNorm1d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))



