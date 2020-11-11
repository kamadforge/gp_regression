
# class BayesianLayer(torch.nn.Module):
#     '''
#     Module implementing a single Bayesian feedforward layer.
#     The module performs Bayes-by-backprop, that is, mean-field
#     variational inference. It keeps prior and posterior weights
#     (and biases) and uses the reparameterization trick for sampling.
#     '''
#     def __init__(self, input_dim, output_dim, bias=True):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.use_bias = bias

#         self.samples_gaussian = None
#         self.samples_gaussian_bias = None

#         # TODO: enter your code here (done)
#         self.prior_mu = nn.Parameter(torch.Tensor(input_dim, output_dim))
#         self.prior_sigma = nn.Parameter(torch.Tensor(input_dim, output_dim))

#         self.weight_mu = nn.Parameter(torch.zeros(input_dim, output_dim).uniform_(-0.2, 0.2))
#         #self.weight_logsigma = nn.Parameter(torch.ones(input_dim, output_dim))
#         self.weight_sigma = nn.Parameter(torch.ones(input_dim, output_dim).uniform_(-5, -4))

#         if self.use_bias:
#             self.bias_mu = nn.Parameter(torch.zeros(output_dim).uniform_(-0.2, 0.2))
#             #self.bias_logsigma = nn.Parameter(torch.zeros(output_dim))
#             self.bias_sigma = nn.Parameter(torch.ones(output_dim).uniform_(-5, -4))
#         else:
#             self.register_parameter('bias_mu', None)
#             #self.register_parameter('bias_logsigma', None)
#             self.register_parameter('bias_sigma', None)

#     def sigma(self, weight_sigma):
#         return torch.log1p(torch.exp(weight_sigma))

#     def kl_divergence(self):
#         '''
#         Computes the KL divergence between the priors and posteriors for this layer.
#         '''
#         kl_loss = self._kl_divergence(self.weight_mu, self.sigma(self.weight_sigma))
#         if self.use_bias:
#             kl_loss += self._kl_divergence(self.bias_mu, self.sigma(self.bias_sigma))
#         return kl_loss

#     def log_prob(self, input, mu, sigma):
#         return (
#             - math.log(math.sqrt(2 * math.pi))
#             - torch.log(sigma)
#             - ((input - mu) ** 2) / (2 * sigma ** 2)
#         ).sum()


#     def _kl_divergence(self, mu, sigma):
#         '''
#         Computes the KL divergence between one Gaussian posterior
#         and the Gaussian prior.
#         '''

#         # TODO: enter your code here (done) tochange: logsigma

#         # you can sample again given mu and sigma, or we keep the samples from the forward pass

#         # 1/sqrt(2*pi*sigma^2) e^{-(x-mu)^2 / 2*sigma^2}
#         # ipsh()

#         self.log_prior = self.log_prob(self.samples_gaussian, self.prior_mu, self.prior_sigma)
#         self.log_variational_posterior = self.log_prob(self.samples_gaussian, mu, sigma)
#         kl = self.log_variational_posterior - self.log_prior

#         # KL:
#         #sum q * log q/p
#         #log q - log p

#         return kl

#     def forward(self, inputs): #logsigma ?
#         samples_stdnormal = torch.empty(self.input_dim, self.output_dim).normal_(mean=0,std=1)
#         self.samples_gaussian = samples_stdnormal * self.sigma(self.weight_sigma) + self.weight_mu

#         if self.use_bias:
#             samples_stdnormal_bias = torch.empty(self.output_dim).normal_(mean=0, std=1)
#             self.samples_gaussian_bias = samples_stdnormal_bias * self.sigma(self.bias_sigma) + self.bias_mu
#         else:
#             bias = None


#         # TODO: enter your code here (done)

#         #print(torch.mean(self.samples_gaussian))
#         output = F.linear(inputs, self.samples_gaussian.t(), self.samples_gaussian_bias)
#         #print(output)
#         return output_