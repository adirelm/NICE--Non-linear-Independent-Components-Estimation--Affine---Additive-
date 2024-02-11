import torch
from tqdm import tqdm
import torchvision.utils
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST

from nice import NICE, StandardLogisticDistribution

def training(normalizing_flow, optimizer, train_loader, test_loader, logistic_distribution, nb_epochs=1500, device='cpu'):
    """
    Trains a normalizing flow model.

    Parameters:
        normalizing_flow (nn.Module): The normalizing flow model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the test data.
        logistic_distribution (StandardLogisticDistribution): The base distribution for the model.
        nb_epochs (int): Number of epochs to train the model.
        device (str): Device to train the model on ('cpu' or 'cuda').

    Returns:
        None
    """
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(nb_epochs)):
        normalizing_flow.train()  # Switch to training mode
        train_loss = 0.0

        # Training loop
        for batch_idx, (batch, _) in enumerate(tqdm(train_loader, leave=False, desc='Training Batch')):
            batch = batch.view(batch.size(0), -1).to(device)  # Flatten and transfer to device
            z, log_jacobian = normalizing_flow(batch) # Forward pass
            log_likelihood = logistic_distribution.log_pdf(z) + log_jacobian
            loss = -log_likelihood.sum()  # Aggregate loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Evaluation loop
        normalizing_flow.eval()  # Switch to evaluation mode
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

        # Logging
        print(f'Epoch {epoch+1}: Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        # Image generation for visualization
        normalizing_flow.eval()  # Set to inference mode
        with torch.no_grad():
            samples = normalizing_flow.invert(logistic_distribution.sample((100,))).cpu()
            # Normalize samples to be in the range [0, 1]
            a, b = samples.min(), samples.max()
            samples = (samples - a) / (b - a + 1e-10)  # Normalize samples to [0, 1]
            samples = samples.view(-1, 1, 28, 28)  # Reshape for MNIST/FashionMNIST
            torchvision.utils.save_image(torchvision.utils.make_grid(samples, nrow=10),
                                         f'Imgs/Generated_{train_loader.dataset.__class__.__name__}_epoch_{epoch}.png')

        normalizing_flow.train()  # Set back to training mode

        # Loss plotting
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
    """
    Loads the specified dataset using torchvision.

    Parameters:
        dataset_name (str): Name of the dataset ('MNIST' or 'FashionMNIST').
        transform (torchvision.transforms): Transformations to apply to the dataset.

    Returns:
        Tuple[Dataset, Dataset]: The training and test datasets.
    """
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

main(dataset_name='FashionMNIST', coupling='additive')