import torch
import torch.nn as nn

from torch.distributions.uniform import Uniform
from torch.distributions.transforms import AffineTransform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution

torch.manual_seed(0)

class StandardLogisticDistribution:
    """
    A class representing the standard logistic distribution, constructed through
    the transformation of a uniform distribution. This class facilitates sampling
    from the logistic distribution and computing the log probability density function
    (log PDF) of samples.

    Attributes:
        m (TransformedDistribution): The transformed distribution representing
        the logistic distribution, achieved by applying a sigmoid transformation
        and an affine transformation to a uniform distribution.

    Parameters:
        data_dim (int): The dimensionality of the data or samples to be generated.
                        Defaults to 28 * 28, suitable for flattened MNIST images.
        device (str): The device on which tensors should be allocated. Defaults
                      to 'cpu'. Can be set to 'cuda' for GPU acceleration.
    """

    def __init__(self, data_dim=28 * 28, device='cpu'):
        """
        Initializes the StandardLogisticDistribution with a uniform distribution
        and applies transformations to create a logistic distribution.

        The transformations include an inverse sigmoid (logit) followed by an
        affine transformation, effectively parameterizing the uniform distribution
        to model a logistic distribution.
        """
        self.m = TransformedDistribution(
            Uniform(torch.zeros(data_dim, device=device),
                    torch.ones(data_dim, device=device)),
            [SigmoidTransform().inv, AffineTransform(torch.zeros(data_dim, device=device),
                                                     torch.ones(data_dim, device=device))])

    def log_pdf(self, z):
        """
        Computes the log probability density function (log PDF) of the given samples.

        Parameters:
            z (Tensor): A tensor containing samples for which to compute the log PDF.

        Returns:
            Tensor: A tensor containing the log PDF values of the input samples, summed
                    across the data dimensions. This sum operation enables compatibility
                    with models that operate on log probabilities as scalars.
        """
        return self.m.log_prob(z).sum(dim=1)

    def sample(self, sample_shape=torch.Size()):
        """
        Generates samples from the logistic distribution.

        Parameters:
            sample_shape (torch.Size, optional): The shape of the sample to generate.
                                                 Defaults to an empty torch.Size(), which
                                                 generates a single sample.

        Returns:
            Tensor: A tensor of samples from the logistic distribution with the specified shape.
        """
        return self.m.sample(sample_shape)


class AdditiveCoupling(nn.Module):
    """
    Implements an additive coupling layer for use in a normalizing flow model.
    This layer partitions the input tensor and applies a transformation to one part
    while leaving the other part unchanged. The transformation consists of a series
    of fully connected layers with ReLU activations.

    Attributes:
        m (torch.nn.ModuleList): A list of sequential models, each representing the
        transformation to be applied to one part of the input tensor.
        s (torch.nn.Parameter): A learnable parameter that scales the output of the
        coupling layer, contributing to the log-Jacobian determinant.

    Parameters:
        data_dim (int): The dimensionality of the input data.
        hidden_dim (int): The dimensionality of the hidden layers within the transformation.
        number_of_hidden (int): The number of hidden layers within each transformation.
        number_of_coupling (int): The number of coupling transformations to apply.
    """
    def __init__(self, data_dim, hidden_dim, number_of_hidden, number_of_coupling):
        super().__init__()
        self.m = torch.nn.ModuleList()
        for _ in range(number_of_coupling):
            # Initialize the transformation for each coupling operation
            layers = [nn.Linear(data_dim // 2, hidden_dim), nn.ReLU()]
            for _ in range(number_of_hidden): # Create the specified number of hidden layers
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, data_dim // 2)) # Output layer to match the partitioned input size
            self.m.append(nn.Sequential(*layers))
        self.s = torch.nn.Parameter(torch.randn(data_dim)) # Scaling parameter for output

    
    def forward(self, x):
        """
        Forward pass of the additive coupling layer.

        Parameters:
            x (Tensor): The input tensor to the coupling layer.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the transformed tensor and the log-Jacobian determinant.
        """
        x = x.clone()
        for i in range(len(self.m)):
            # Partition the input tensor and apply the transformation to one part
            x_i1 = x[:, ::2] if (i % 2) == 0 else x[:, 1::2]
            x_i2 = x[:, 1::2] if (i % 2) == 0 else x[:, ::2]
            h_i1 = x_i1
            h_i2 = x_i2 + self.m[i](x_i1) # Apply transformation
            x = torch.empty(x.shape, device=x.device) # Prepare a tensor for the output
            x[:, ::2] = h_i1
            x[:, 1::2] = h_i2
        z = torch.exp(self.s) * x # Scale the transformed tensor
        log_jacobian = torch.sum(self.s) # Compute log-Jacobian determinant
        return z, log_jacobian

    def inverse(self, z):
        """
        Inverse pass of the additive coupling layer, reversing the transformation.

        Parameters:
            z (Tensor): The transformed tensor for which to compute the inverse transformation.

        Returns:
            Tensor: The original input tensor before transformation.
        """
        x = z.clone() / torch.exp(self.s)
        for i in range(len(self.m) - 1, -1, -1):
            # Reverse the transformation applied during the forward pass
            h_i1 = x[:, ::2]
            h_i2 = x[:, 1::2]
            x_i1 = h_i1
            x_i2 = h_i2 - self.m[i](x_i1) # Apply inverse transformation
            x = torch.empty(x.shape, device=x.device) # Prepare a tensor for the output
            x[:, ::2] = x_i1 if (i % 2) == 0 else x_i2
            x[:, 1::2] = x_i2 if (i % 2) == 0 else x_i1
        return x

class AffineCoupling(nn.Module):
    """
    Implements an affine coupling layer for use in a normalizing flow model.
    This layer splits the input tensor and transforms one part by scaling and
    translating it, based on functions of the other part. This transformation
    allows the model to learn more complex distributions in a reversible manner,
    crucial for normalizing flow models.

    Attributes:
        m (torch.nn.ModuleList): A list of sequential neural networks, where each
        network computes the translation (shift) parameters for one part of the
        input tensor based on the other part.
        s (torch.nn.Parameter): A learnable parameter vector that provides scaling
        factors for the affine transformation, applied to half of the input dimensions.

    Parameters:
        data_dim (int): The total dimensionality of the input data.
        hidden_dim (int): The number of units in the hidden layers of the neural networks.
        number_of_hidden (int): The number of hidden layers in each neural network.
        number_of_coupling (int): The number of sequential affine coupling layers to apply.
    """
    def __init__(self, data_dim, hidden_dim, number_of_hidden, number_of_coupling):
        super().__init__()
        self.m = torch.nn.ModuleList()
        for _ in range(number_of_coupling):
            # Initialize each neural network for computing translation parameters
            layers = [nn.Linear(data_dim // 2, hidden_dim), nn.ReLU()]
            for _ in range(number_of_hidden):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, data_dim // 2))
            self.m.append(nn.Sequential(*layers))
        self.s = torch.nn.Parameter(torch.randn(data_dim // 2))  # Scaling parameters

    def forward(self, x):
        """
        Forward pass through the affine coupling layer.

        Parameters:
            x (Tensor): The input tensor to the coupling layer.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the transformed tensor and
            the log-Jacobian determinant of the transformation.
        """
        x = x.clone() # Avoid modifying input in-place
        for i in range(len(self.m)):
            # Split input tensor and apply affine transformation to one part
            x_i1 = x[:, :x.size(1) // 2] if (i % 2) == 0 else x[:, x.size(1) // 2:]
            x_i2 = x[:, x.size(1) // 2:] if (i % 2) == 0 else x[:, :x.size(1) // 2]
            t_i = self.m[i](x_i1) # Compute translatio
            x_i2 = x_i2 + t_i # Apply translation
            # Reassemble the tensor
            if (i % 2) == 0:
                x = torch.cat((x_i1, x_i2), dim=1)
            else:
                x = torch.cat((x_i2, x_i1), dim=1)
        # Apply scaling
        s_expanded = torch.exp(self.s).repeat(2) 
        z = x * s_expanded
        log_jacobian = self.s.sum() * 2  # Compute log-Jacobian
        return z, log_jacobian

    def inverse(self, z):
        """
        Inverse transformation through the affine coupling layer.

        Parameters:
            z (Tensor): The transformed tensor for which to compute the inverse.

        Returns:
            Tensor: The original input tensor before the affine transformation.
        """
        # Apply inverse scaling
        s_expanded = torch.exp(self.s).repeat(2)  
        x = z / s_expanded
        for i in reversed(range(len(self.m))):
            # Reverse the affine transformation
            x_i1 = x[:, :x.size(1) // 2] if (i % 2) == 0 else x[:, x.size(1) // 2:]
            x_i2 = x[:, x.size(1) // 2:] if (i % 2) == 0 else x[:, :x.size(1) // 2]
            t_i = self.m[i](x_i1) # Compute translation
            x_i2 = x_i2 - t_i # Reverse translation
            # Reassemble the tensor
            if (i % 2) == 0:
                x = torch.cat((x_i1, x_i2), dim=1)
            else:
                x = torch.cat((x_i2, x_i1), dim=1)
        return x



class NICE(nn.Module):
    """
    Implements the NICE (Non-linear Independent Components Estimation) model, a type of
    normalizing flow for density estimation and generative modeling. The model uses
    coupling layers (either additive or affine) to perform invertible transformations,
    allowing for efficient computation of the likelihood of the data and sampling from
    the model.

    Attributes:
        coupling (str): The type of coupling used in the model ('additive' or 'affine').
        network (nn.Module): The coupling layer network, either AdditiveCoupling or
                             AffineCoupling, based on the specified coupling type.

    Parameters:
        data_dim (int): The dimensionality of the input data. Defaults to 28 * 28,
                        suitable for flattened MNIST images.
        hidden_dim (int): The number of units in the hidden layers of the coupling layers.
                          Defaults to 1000.
        number_of_hidden (int): The number of hidden layers within each segment of the
                                coupling layers. Defaults to 4.
        coupling (str): Specifies the type of coupling layer to use. Can be 'additive'
                        for additive coupling layers or 'affine' for affine coupling layers.
                        Defaults to 'additive'.
        number_of_coupling (int): The number of coupling layers to include in the model.
                                  Defaults to 4.
    """
    def __init__(self, data_dim=28 * 28, hidden_dim=1000, number_of_hidden = 4, coupling='additive', number_of_coupling = 4):
        super().__init__()
        self.coupling = coupling
        # Initialize the appropriate coupling layer based on the coupling type
        if self.coupling == 'additive':
            self.network = AdditiveCoupling(data_dim, hidden_dim, number_of_hidden, number_of_coupling)
        elif self.coupling == 'affine':
            self.network = AffineCoupling(data_dim, hidden_dim, number_of_hidden, number_of_coupling)

    def forward(self, x):
        """
        Forward pass through the NICE model. Applies the coupling layer transformation
        to the input data and returns the transformed data along with the log-Jacobian
        determinant of the transformation.

        Parameters:
            x (Tensor): The input tensor to the model.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the transformed tensor (z) and
            the log-Jacobian determinant of the transformation.
        """
        z, log_jacobian = self.network(x)
        return z, log_jacobian

    def invert(self, z):
        """
        Inverts the transformation applied by the NICE model, recovering the original
        input data from the transformed data.

        Parameters:
            z (Tensor): The transformed tensor for which to compute the inverse transformation.

        Returns:
            Tensor: The original input tensor before transformation.
        """
        z = self.network.inverse(z)
        return z