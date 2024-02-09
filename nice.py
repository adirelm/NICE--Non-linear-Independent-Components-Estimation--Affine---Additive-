import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.utils
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from torch.distributions.uniform import Uniform
from torch.distributions.transforms import AffineTransform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution

torch.manual_seed(0)

class StandardLogisticDistribution:

    def __init__(self, data_dim=28 * 28, device='cpu'):
        self.m = TransformedDistribution(
            Uniform(torch.zeros(data_dim, device=device),
                    torch.ones(data_dim, device=device)),
            [SigmoidTransform().inv, AffineTransform(torch.zeros(data_dim, device=device),
                                                     torch.ones(data_dim, device=device))]
        )

    def log_pdf(self, z):
        return self.m.log_prob(z).sum(dim=1)

    def sample(self, sample_shape=torch.Size()):
        return self.m.sample(sample_shape)


class AdditiveCoupling(nn.Module):
    def __init__(self, data_dim, hidden_dim, hidden_number, coupling_number):
        super().__init__()
        self.m = torch.nn.ModuleList()
        for _ in range(coupling_number):
            layers = [nn.Linear(data_dim // 2, hidden_dim), nn.ReLU()]
            for _ in range(hidden_number):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, data_dim // 2))
            self.m.append(nn.Sequential(*layers))
        self.s = torch.nn.Parameter(torch.randn(data_dim))

    
    def forward(self, x):
        x = x.clone()
        for i in range(len(self.m)):
            x_i1 = x[:, ::2] if (i % 2) == 0 else x[:, 1::2]
            x_i2 = x[:, 1::2] if (i % 2) == 0 else x[:, ::2]
            h_i1 = x_i1
            h_i2 = x_i2 + self.m[i](x_i1)
            x = torch.empty(x.shape, device=x.device)
            x[:, ::2] = h_i1
            x[:, 1::2] = h_i2
        z = torch.exp(self.s) * x
        log_jacobian = torch.sum(self.s)
        return z, log_jacobian

    def inverse(self, z):
        x = z.clone() / torch.exp(self.s)
        for i in range(len(self.m) - 1, -1, -1):
            h_i1 = x[:, ::2]
            h_i2 = x[:, 1::2]
            x_i1 = h_i1
            x_i2 = h_i2 - self.m[i](x_i1)
            x = torch.empty(x.shape, device=x.device)
            x[:, ::2] = x_i1 if (i % 2) == 0 else x_i2
            x[:, 1::2] = x_i2 if (i % 2) == 0 else x_i1
        return x

class NICE(nn.Module):
    def __init__(self, data_dim=28 * 28, hidden_dim=1000, hidden_number = 4, coupling='additive', coupling_number = 4):
        super().__init__()
        self.coupling = coupling
        if self.coupling == 'additive':
            self.network = AdditiveCoupling(data_dim, hidden_dim, hidden_number, coupling_number)

    def forward(self, x):
        z, log_jacobian = self.network(x)
        return z, log_jacobian

    def invert(self, z):
        z = self.network.inverse(z)
        return z


def training(normalizing_flow, optimizer, dataloader, distribution, logistic_distribution, nb_epochs=1500, device='cpu'):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        for batch_idx, (batch, _) in enumerate(dataloader):
            batch = batch.view(batch.size(0), -1).to(device)  # Flatten and move to the correct device
            z, log_jacobian = normalizing_flow(batch)
            log_likelihood = logistic_distribution.log_pdf(z) + log_jacobian
            loss = -log_likelihood.sum()  # Calculate mean loss over batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

        # Generate and save images after each epoch
        normalizing_flow.eval()  # Set to inference mode
        with torch.no_grad():
            samples = normalizing_flow.invert(logistic_distribution.sample((100,))).cpu()
            a, b = samples.min(), samples.max()
            samples = (samples - a) / (b - a + 1e-10)  # Normalize samples to [0, 1]
            samples = samples.view(-1, 1, 28, 28)  # Assuming MNIST, reshape to BxCxHxW format
            torchvision.utils.save_image(torchvision.utils.make_grid(samples, nrow=10),
                                         f'Imgs/Generated_MNIST_epoch_{epoch}.png')

        normalizing_flow.train()  # Set back to training mode

    return training_loss


def main():
    device = 'cpu'
    normalizing_flow = NICE().to(device)
    logistic_distribution = StandardLogisticDistribution(device=device)
    
    # MNIST Data loading with torchvision
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize images
    ])
    
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(normalizing_flow.parameters(), lr=0.0002, weight_decay=0.9)
    
    training_loss = training(normalizing_flow, optimizer, train_loader, logistic_distribution, nb_epochs=500,
                             device=device, logistic_distribution=logistic_distribution)
                             

    nb_data = 10
    fig, axs = plt.subplots(nb_data, nb_data, figsize=(10, 10))
    for i in range(nb_data):
        for j in range(nb_data):
            x = normalizing_flow.invert(logistic_distribution.sample().unsqueeze(0)).data.cpu().numpy()
            axs[i, j].imshow(x.reshape(28, 28).clip(0, 1), cmap='gray')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    plt.savefig('Imgs/Generated_MNIST_data.png')
    plt.show()

main()