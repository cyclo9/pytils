from gpflow.models import SVGP
from gpflow.kernels import RBF, Matern12
from gpflow.likelihoods import Gaussian
from gpflow.inducing_variables import InducingPoints


def create_model(Z):
    kernel = Matern12(lengthscales=0.75)
    likelihood = Gaussian()
    inducing_points = InducingPoints(Z)
    model = SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing_points,
        num_latent_gps=1,
    )
    return model


# import gpytorch
# from gpytorch.models import ApproximateGP
# from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
# from gpytorch.kernels import RBFKernel
#
#
# class SparseGP(ApproximateGP):
#     def __init__(self, inducing_points):
#         variational_dist = CholeskyVariationalDistribution(inducing_points.size(0))
#         variational_strategy = VariationalStrategy(
#             self, inducing_points, variational_dist, learn_inducing_locations=True
#         )
#         super().__init__(variational_strategy)
#         self.mean_module = gpytorch.means.ZeroMean()
#         self.covar_module = RBFKernel()
#
#     def forward(self, x):
#         mean = self.mean_module(x)
#         covar = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean, covar)
