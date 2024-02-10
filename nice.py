import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.utils
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
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
    def __init__(self, data_dim, hidden_dim, number_of_hidden, number_of_coupling):
        super().__init__()
        self.m = torch.nn.ModuleList()
        for _ in range(number_of_coupling):
            layers = [nn.Linear(data_dim // 2, hidden_dim), nn.ReLU()]
            for _ in range(number_of_hidden):
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

class AffineCoupling(nn.Module):
    def __init__(self, data_dim, hidden_dim, number_of_hidden, number_of_coupling):
        super().__init__()
        self.m = torch.nn.ModuleList()
        for _ in range(number_of_coupling):
            layers = [nn.Linear(data_dim // 2, hidden_dim), nn.ReLU()]
            for _ in range(number_of_hidden):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, data_dim // 2))  # Ensures output dimension is correct for translation
            self.m.append(nn.Sequential(*layers))
        self.s = torch.nn.Parameter(torch.randn(data_dim // 2))  # Corrected to match dimensions for scaling

    def forward(self, x):
        x = x.clone()
        for i in range(len(self.m)):
            x_i1 = x[:, :x.size(1) // 2] if (i % 2) == 0 else x[:, x.size(1) // 2:]
            x_i2 = x[:, x.size(1) // 2:] if (i % 2) == 0 else x[:, :x.size(1) // 2]
            t_i = self.m[i](x_i1)
            x_i2 = x_i2 + t_i
            if (i % 2) == 0:
                x = torch.cat((x_i1, x_i2), dim=1)
            else:
                x = torch.cat((x_i2, x_i1), dim=1)
        # Correctly apply scaling
        s_expanded = torch.exp(self.s).repeat(2)  # Duplicate the scaling to match the full dimension
        z = x * s_expanded
        log_jacobian = self.s.sum() * 2  # Adjusted for the duplicated scaling factors
        return z, log_jacobian

    def inverse(self, z):
        # Correctly apply inverse scaling
        s_expanded = torch.exp(self.s).repeat(2)  # Duplicate the scaling to match the full dimension
        x = z / s_expanded  # Apply the corrected inverse scaling
        for i in reversed(range(len(self.m))):
            x_i1 = x[:, :x.size(1) // 2] if (i % 2) == 0 else x[:, x.size(1) // 2:]
            x_i2 = x[:, x.size(1) // 2:] if (i % 2) == 0 else x[:, :x.size(1) // 2]
            t_i = self.m[i](x_i1)
            x_i2 = x_i2 - t_i
            if (i % 2) == 0:
                x = torch.cat((x_i1, x_i2), dim=1)
            else:
                x = torch.cat((x_i2, x_i1), dim=1)
        return x



class NICE(nn.Module):
    def __init__(self, data_dim=28 * 28, hidden_dim=1000, number_of_hidden = 4, coupling='additive', number_of_coupling = 4):
        super().__init__()
        self.coupling = coupling
        if self.coupling == 'additive':
            self.network = AdditiveCoupling(data_dim, hidden_dim, number_of_hidden, number_of_coupling)
        elif self.coupling == 'affine':
            self.network = AffineCoupling(data_dim, hidden_dim, number_of_hidden, number_of_coupling)

    def forward(self, x):
        z, log_jacobian = self.network(x)
        return z, log_jacobian

    def invert(self, z):
        z = self.network.inverse(z)
        return z


def training(normalizing_flow, optimizer, train_loader, test_loader, logistic_distribution, nb_epochs=1500, device='cpu'):
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(nb_epochs)):
        normalizing_flow.train()  # Ensure the model is in training mode
        train_loss = 0.0

        # Training phase
        for batch_idx, (batch, _) in enumerate(tqdm(train_loader, leave=False, desc='Training Batch')):
            batch = batch.view(batch.size(0), -1).to(device)  # Flatten and move to the correct device
            z, log_jacobian = normalizing_flow(batch)
            log_likelihood = logistic_distribution.log_pdf(z) + log_jacobian
            loss = -log_likelihood.sum()  # Aggregate loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Evaluation phase for test loss
        normalizing_flow.eval()  # Set the model to evaluation mode
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (batch, _) in enumerate(tqdm(test_loader, leave=False, desc='Test Batch')):
                batch = batch.view(batch.size(0), -1).to(device)
                z, log_jacobian = normalizing_flow(batch)
                log_likelihood = logistic_distribution.log_pdf(z) + log_jacobian
                loss = -log_likelihood.sum()
                test_loss += loss.item()

            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)

        print(f'Epoch {epoch+1}: Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        # Generate and save images after each epoch
        normalizing_flow.eval()  # Set to inference mode
        with torch.no_grad():
            samples = normalizing_flow.invert(logistic_distribution.sample((100,))).cpu()
            a, b = samples.min(), samples.max()
            samples = (samples - a) / (b - a + 1e-10)  # Normalize samples to [0, 1]
            samples = samples.view(-1, 1, 28, 28)  # Assuming MNIST, reshape to BxCxHxW format
            torchvision.utils.save_image(torchvision.utils.make_grid(samples, nrow=10),
                                         f'Imgs/Generated_{train_loader.dataset.__class__.__name__}_epoch_{epoch}.png')

        normalizing_flow.train()  # Set back to training mode

        # Plot and save losses after each epoch
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 2), train_losses, label='Training Loss', marker='o')
        plt.plot(range(1, epoch + 2), test_losses, label='Test Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log Likelihood')
        plt.title(f'Training and Test Loss - Epoch {epoch + 1}')
        plt.legend()
        plt.savefig(f'Plots/Loss_Epoch_{epoch + 1}.png')  # Save the figure
        plt.close()  # Close the figure to free memory

def load_dataset(dataset_name, transform):
    if dataset_name == 'MNIST':
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset. Choose 'MNIST' or 'FashionMNIST'.")
    return train_dataset, test_dataset

def main(dataset_name='MNIST', coupling = 'additive', hidden_dim = 1000, number_of_hidden = 4, number_of_coupling = 4):
    device = 'cpu'

    normalizing_flow = NICE(
        coupling = coupling,
        hidden_dim = hidden_dim,
        number_of_hidden = number_of_hidden,
        number_of_coupling = number_of_coupling).to(device)
    
    logistic_distribution = StandardLogisticDistribution(device=device)
    
    # MNIST Data loading with torchvision
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize images
    ])
    
    train_dataset, test_dataset = load_dataset(dataset_name, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(normalizing_flow.parameters(), lr=0.0002, weight_decay=0.9)
    
    training(normalizing_flow, optimizer, train_loader, test_loader, nb_epochs=50,
                             device=device, logistic_distribution=logistic_distribution)

main(dataset_name='MNIST', coupling='additive')