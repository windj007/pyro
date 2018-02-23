from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Function, variable
from torch.autograd.function import once_differentiable
from torch.distributions import constraints

from pyro.distributions.torch.multivariate_normal import MultivariateNormal
from pyro.distributions.util import sum_leftmost


class OMTMultivariateNormal(MultivariateNormal):
    """Multivariate normal (Gaussian) distribution with OMT gradients w.r.t. both
    parameters. Note the gradient computation w.r.t. the Cholesky factor has cost
    O(D^3), although the resulting gradient variance is generally expected to be lower.

    A distribution over vectors in which all the elements have a joint Gaussian
    density.

    :param torch.autograd.Variable loc: Mean.
    :param torch.autograd.Variable scale_tril: Cholesky of Covariance matrix.
    """
    params = {"loc": constraints.real, "scale_tril": constraints.lower_triangular}

    def __init__(self, loc, scale_tril):
        assert(loc.dim() == 1), "OMTMultivariateNormal loc must be 1-dimensional"
        assert(scale_tril.dim() == 2), "OMTMultivariateNormal scale_tril must be 2-dimensional"
        covariance_matrix = torch.mm(scale_tril, scale_tril.t())
        super(OMTMultivariateNormal, self).__init__(loc, covariance_matrix)
        self.scale_tril = scale_tril

    def rsample(self, sample_shape=torch.Size()):
        return _OMTMVNSample.apply(self.loc, self.scale_tril, sample_shape + self.loc.shape)


class _OMTMVNSample(Function):
    @staticmethod
    def forward(ctx, loc, scale_tril, shape):
        ctx.white = loc.new(shape).normal_()
        ctx.z = torch.matmul(ctx.white, scale_tril.t())
        ctx.save_for_backward(scale_tril)
        return loc + ctx.z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        jitter = 1.0e-8  # do i really need this?
        L, = ctx.saved_tensors
        z = ctx.z
        epsilon = ctx.white

        dim = L.size(0)
        g = grad_output
        loc_grad = sum_leftmost(grad_output, -1)

        identity = torch.eye(dim, out=variable(g.new(dim, dim)))
        R_inv = torch.trtrs(identity, L.t(), transpose=False, upper=True)[0]

        z_ja = z.unsqueeze(-1)
        g_R_inv = torch.matmul(g, R_inv).unsqueeze(-2)
        epsilon_jb = epsilon.unsqueeze(-2)
        g_ja = g.unsqueeze(-1)
        diff_L_ab = 0.5 * sum_leftmost(g_ja * epsilon_jb + g_R_inv * z_ja, -2)

        Sigma_inv = torch.mm(R_inv, R_inv.t())
        V, D, _ = torch.svd(Sigma_inv + jitter)
        D_outer = D.unsqueeze(-1) + D.unsqueeze(0)

        expand_tuple = tuple([-1] * (z.dim() - 1) + [dim, dim])
        z_tilde = identity * torch.matmul(z, V).unsqueeze(-1).expand(*expand_tuple)
        g_tilde = identity * torch.matmul(g, V).unsqueeze(-1).expand(*expand_tuple)

        Y = sum_leftmost(torch.matmul(z_tilde, torch.matmul(1.0 / D_outer, g_tilde)), -2)
        Y = torch.mm(V, torch.mm(Y, V.t()))
        Y = Y + Y.t()

        Tr_xi_Y = torch.mm(torch.mm(Sigma_inv, Y), R_inv) - torch.mm(Y, torch.mm(Sigma_inv, R_inv))
        diff_L_ab += 0.5 * Tr_xi_Y
        L_grad = torch.tril(diff_L_ab)

        return loc_grad, L_grad, None


class OTCVMultivariateNormal(MultivariateNormal):
    """Multivariate normal (Gaussian) distribution with optimal transport-inspired control variates.

    A distribution over vectors in which all the elements have a joint Gaussian
    density.

    :param torch.autograd.Variable loc: Mean.
    :param torch.autograd.Variable scale_tril: Cholesky of Covariance matrix.
    :param torch.autograd.Variable B: tensor controlling the control variate
    :param torch.autograd.Variable C: tensor controlling the control variate
    :param torch.autograd.Variable D: tensor controlling the control variate
    :param torch.autograd.Variable F: tensor controlling the control variate
    """
    params = {"loc": constraints.real, "scale_tril": constraints.lower_triangular}

    def __init__(self, loc, scale_tril, CV=None):
        assert(loc.dim() == 1), "OMTMultivariateNormal loc must be 1-dimensional"
        assert(scale_tril.dim() == 2), "OMTMultivariateNormal scale_tril must be 2-dimensional"
        covariance_matrix = torch.mm(scale_tril, scale_tril.t())
        super(OTCVMultivariateNormal, self).__init__(loc, covariance_matrix)
        self.scale_tril = scale_tril
        self.CV = CV
        assert(CV.size() == torch.Size([2, loc.size(0), loc.size(0)]))

    def rsample(self, sample_shape=torch.Size()):
        return _OTCVMVNSample.apply(self.loc, self.scale_tril, self.CV, sample_shape + self.loc.shape)


class _OTCVMVNSample(Function):
    @staticmethod
    def forward(ctx, loc, scale_tril, CV, shape):
        ctx.save_for_backward(scale_tril)
        ctx.white = loc.new(shape).normal_()
        ctx.z = torch.matmul(ctx.white, scale_tril.t())
        ctx.CV = CV
        return loc + ctx.z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        lr = ctx.lr
        L, = ctx.saved_tensors
        z = ctx.z
        epsilon = ctx.white
        B, C = ctx.CV

        dim = L.size(0)
        g = grad_output
        loc_grad = sum_leftmost(grad_output, -1)

        # compute the rep trick gradient
        epsilon_jb = epsilon.unsqueeze(-2)
        g_ja = g.unsqueeze(-1)
        diff_L_ab = sum_leftmost(g_ja * epsilon_jb, -2)

        # modulate the velocity fields with infinitessimal rotations, i.e. apply the control variate
            LB = torch.mm(L, B)
            eps_C = torch.matmul(epsilon, C)
            g_LB = torch.matmul(g, LB)
            diff_L_ab += sum_leftmost(eps_C.unsqueeze(-2) * g_LB.unsqueeze(-1), -2)
            LC = torch.mm(L, C)
            eps_B = torch.matmul(epsilon, B)
            g_LC = torch.matmul(g, LC)
            diff_L_ab -= sum_leftmost(eps_B.unsqueeze(-1) * g_LC.unsqueeze(-2), -2)

        L_grad = torch.tril(diff_L_ab)

        # adapt B and C
        if BC_mode:
            g_L_kx = torch.matmul(g, L).unsqueeze(-1)
            dL_C_eps_ky = torch.matmul(eps_C, L_grad.t()).unsqueeze(-2)
            dB_xy = 2.0 * sum_leftmost(g_L_kx * dL_C_eps_ky, -2)
            g_LC_y = torch.matmul(torch.matmul(g, LC), L_grad.t()).unsqueeze(-2)
            eps_x = epsilon.unsqueeze(-1)
            dB_xy -= 2.0 * sum_leftmost(g_LC_y * eps_x, -2)

            g_LB_y = torch.matmul(g, torch.mm(LB, L_grad)).unsqueeze(-2)
            dC_xy = 2.0 * sum_leftmost(g_LB_y * eps_x, -2)
            eps_BLg_y = torch.matmul(epsilon, torch.mm(B, L_grad)).unsqueeze(-2)
            dC_xy -= 2.0 * sum_leftmost(eps_BLg_y * g_L_kx, -2)


        return loc_grad, L_grad, dB_xy, dC_xy, None, None, None, None


class OldOTCVMultivariateNormal(MultivariateNormal):
    """Multivariate normal (Gaussian) distribution with optimal transport-inspired control variates.

    A distribution over vectors in which all the elements have a joint Gaussian
    density.

    :param torch.autograd.Variable loc: Mean.
    :param torch.autograd.Variable scale_tril: Cholesky of Covariance matrix.
    :param torch.autograd.Variable B: tensor controlling the control variate
    :param torch.autograd.Variable C: tensor controlling the control variate
    :param torch.autograd.Variable D: tensor controlling the control variate
    :param torch.autograd.Variable F: tensor controlling the control variate
    """
    params = {"loc": constraints.real, "scale_tril": constraints.lower_triangular}

    def __init__(self, loc, scale_tril, B=None, C=None, D=None, F=None, lr=0.01):
        assert(loc.dim() == 1), "OMTMultivariateNormal loc must be 1-dimensional"
        assert(scale_tril.dim() == 2), "OMTMultivariateNormal scale_tril must be 2-dimensional"
        covariance_matrix = torch.mm(scale_tril, scale_tril.t())
        super(OTCVMultivariateNormal, self).__init__(loc, covariance_matrix)
        self.scale_tril = scale_tril
        self.B = B
        self.C = C
        self.D = D
        self.F = F
        self.lr = lr
        BC_mode = (B is not None) and (C is not None)
        DF_mode = (D is not None) and (F is not None)
        if BC_mode:
            assert(scale_tril.size() == B.size() == C.size())
        if DF_mode:
            assert(scale_tril.size() == D.size() == F.size())
        assert (BC_mode or DF_mode), "Must use at least one control variate parameterization"

    def rsample(self, sample_shape=torch.Size()):
        return _OldOTCVMVNSample.apply(self.loc, self.scale_tril, self.B, self.C, self.D, self.F,
                                    sample_shape + self.loc.shape, self.lr)


class _OldOTCVMVNSample(Function):
    @staticmethod
    def forward(ctx, loc, scale_tril, B, C, D, F, shape, lr):
        ctx.save_for_backward(scale_tril)
        ctx.white = loc.new(shape).normal_()
        ctx.z = torch.matmul(ctx.white, scale_tril.t())
        ctx.B, ctx.C = B, C
        ctx.D, ctx.F = D, F
        ctx.lr = lr
        return loc + ctx.z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        lr = ctx.lr
        L, = ctx.saved_tensors
        z = ctx.z
        epsilon = ctx.white
        B, C = ctx.B, ctx.C
        D, F = ctx.D, ctx.F
        BC_mode = (B is not None) and (C is not None)
        DF_mode = (D is not None) and (F is not None)

        dim = L.size(0)
        g = grad_output
        loc_grad = sum_leftmost(grad_output, -1)

        # compute the rep trick gradient
        epsilon_jb = epsilon.unsqueeze(-2)
        g_ja = g.unsqueeze(-1)
        diff_L_ab = sum_leftmost(g_ja * epsilon_jb, -2)

        # modulate the velocity fields with infinitessimal rotations
        if BC_mode:
            LB = torch.mm(L, B)
            eps_C = torch.matmul(epsilon, C)
            g_LB = torch.matmul(g, LB)
            diff_L_ab += sum_leftmost(eps_C.unsqueeze(-2) * g_LB.unsqueeze(-1), -2)
            LC = torch.mm(L, C)
            eps_B = torch.matmul(epsilon, B)
            g_LC = torch.matmul(g, LC)
            diff_L_ab -= sum_leftmost(eps_B.unsqueeze(-1) * g_LC.unsqueeze(-2), -2)

        # modulate the velocity fields with infinitessimal rotations
        if DF_mode:
            LD = torch.mm(L, D)
            g_LD = torch.matmul(g, LD)
            multiplier = (g_LD * epsilon).sum()
            diff_L_ab += multiplier * F

        L_grad = torch.tril(diff_L_ab)

        # adapt B and C
        if BC_mode:
            g_L_kx = torch.matmul(g, L).unsqueeze(-1)
            dL_C_eps_ky = torch.matmul(eps_C, L_grad.t()).unsqueeze(-2)
            dB_xy = 2.0 * sum_leftmost(g_L_kx * dL_C_eps_ky, -2)
            g_LC_y = torch.matmul(torch.matmul(g, LC), L_grad.t()).unsqueeze(-2)
            eps_x = epsilon.unsqueeze(-1)
            dB_xy -= 2.0 * sum_leftmost(g_LC_y * eps_x, -2)

            g_LB_y = torch.matmul(g, torch.mm(LB, L_grad)).unsqueeze(-2)
            dC_xy = 2.0 * sum_leftmost(g_LB_y * eps_x, -2)
            eps_BLg_y = torch.matmul(epsilon, torch.mm(B, L_grad)).unsqueeze(-2)
            dC_xy -= 2.0 * sum_leftmost(eps_BLg_y * g_L_kx, -2)

            #B -= lr * dB_xy
            #C -= lr * dC_xy

        return loc_grad, L_grad, dB_xy, dC_xy, None, None, None, None
