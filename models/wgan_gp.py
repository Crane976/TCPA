# models/wgan_gp.py
import torch
import torch.nn as nn
import torch.autograd as autograd


class ProGenGenerator(nn.Module):
    """
    ProGen 的生成器 (适配 Feature Space)。
    Input: Benign Sample (Real) + Noise (Optional)
    Output: Adversarial/Decoy Sample
    Logic: Projection (ResNet-style residual learning is often better, but pure MLP is standard)
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(ProGenGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出归一化到 [0, 1]
        )

    def forward(self, x):
        return self.net(x)


class ProGenDiscriminator(nn.Module):
    """
    WGAN-GP 的判别器 (Critic)。
    Input: Sample (Real Bot or Generated Decoy)
    Output: Wasserstein Score (Scalar)
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(ProGenDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)  # WGAN 输出不需要 Sigmoid
        )

    def forward(self, x):
        return self.net(x)


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1)).to(device)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates)

    fake = torch.ones((real_samples.size(0), 1), requires_grad=False).to(device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty