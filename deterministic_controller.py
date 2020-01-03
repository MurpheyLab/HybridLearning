import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
from torch.distributions import Normal


def compute_jacobian(inputs, output, create_graph=False):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True, create_graph=create_graph)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)

class HybridDeterControl(object):

    def __init__(self, model, policy, T=10, eps=1e-1, reg=1.0):
        self.model = model
        self.policy = policy
        self.T = T
        self.eps = eps
        self.reg = reg

        self.num_states = model.num_states
        self.num_actions = model.num_actions

        self.u = torch.cat(
            [torch.zeros(1, self.num_actions) for t in range(self.T)], dim=0)
        self.u.requires_grad = True

    def reset(self):
        self.u = torch.cat(
            [torch.zeros(1, self.num_actions) for t in range(self.T)], dim=0)
        self.u.requires_grad = True


    def __call__(self, state):
        with torch.no_grad():
            self.u[:-1] = self.u[1:].clone()
            self.u[-1].zero_()
            x_t = torch.FloatTensor(state).unsqueeze(0)
            x = []
            u = []
            for t in range(self.T):
                x.append(x_t.clone())
                u_t, log_std_t = self.policy(x_t)
                # x_t, r_t = self.model.step(x_t, self.u[t].unsqueeze(0) + u_t)
                x_t, r_t = self.model.step(x_t, u_t)
                u.append(u_t.clone())

        # compute those derivatives
        x = torch.cat(x)
        x.requires_grad = True
        u = torch.cat(u)
        u.requires_grad = True

        u_p, log_std_p = self.policy(x)

        # pi = Normal(u_p, log_std_p.exp())

        pred_state, pred_rew = self.model.step(x, u_p)
        _ps, _pr = self.model.step(x.detach(), u)
        # pred_rew = pred_rew + torch.mean(torch.pow(self.u, 2))

        # pred_state, pred_rew = self.model.step(x, torch.tanh(self.u+u_p))
        # log_prob = pi.log_prob(self.u+u_p)
        # log_prob = log_prob - torch.max(log_prob, dim=1, keepdim=True)[0]

        # pred_rew = pred_rew + torch.sum(log_prob, dim=1, keepdim=True)


        # loss = pred_rew.mean()
        dfdx = compute_jacobian(x, pred_state)
        dfdu = compute_jacobian(u, _ps)
        dldx = compute_jacobian(x, pred_rew)
        # dldu = compute_jacobian(self.u, torch.sum(log_prob, dim=1, keepdim=True))

        # dfdx = dfdx/(torch.norm(dfdx, dim=[1,2],keepdim=True)+1e-4)
        # dfdu = dfdu/(torch.norm(dfdu, dim=[1,2],keepdim=True)+1e-4)
        # dl = torch.cat([dldx, dldu], dim=2)
        # dl_norm = torch.norm(dl, dim=[1,2],keepdim=True)+1e-4
        dldx = dldx/(torch.norm(dldx, dim=[1,2],keepdim=True)+1e-4)
        # dldu = dldu/(torch.norm(dldu, dim=[1,2],keepdim=True)+1e-4)

        with torch.no_grad():
            rho = torch.zeros(1, self.num_states)
            for t in reversed(range(self.T)):
                rho = dldx[t] + rho.mm(dfdx[t])
                # self.u[t] = self.u[t] + (dldu[t] + rho.mm(dfdu[t])) * self.eps
                # self.u[t] = 2.0* log_std_p[t].exp() * rho.mm(dfdu[t])
            _u = 2.0 * log_std_p[0].exp() * rho.mm(dfdu[0])
        f1,_ = self.model.step(x[0].unsqueeze(0), torch.clamp(u_p[0].unsqueeze(0),-1,+1) )
        f2,_ = self.model.step(x[0].unsqueeze(0), torch.clamp(_u[0]+u_p[0].unsqueeze(0),-1,+1))
        return torch.clamp(_u[0]+u_p[0],-1,+1).detach().clone().numpy(), rho.mm((f2-f1).T).detach().clone().numpy().squeeze()

        # return torch.tanh(self.u[0] + u_p[0]).detach().clone().numpy()
