import os
import torch
from torch import nn
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from opts import Lion

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(32 * 32 * 3, 64),
      nn.Sigmoid(),
      nn.Linear(64, 32),
      nn.Sigmoid(),
      nn.Linear(32, 10)
    )
    
  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
  
  
if __name__ == '__main__':
  
    # Set fixed random number seed
    torch.manual_seed(42)

    # Prepare CIFAR-10 dataset
    # dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=1)

    # Initialize the MLP
    mlp = MLP()

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    # Adam
    # optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    optimizer = Lion(mlp.parameters(), lr=3e-5, weight_decay=10.)
    # Run the training loop
    losses = []
    for epoch in range(0, 100): # 5 epochs at maximum

    # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            
            # Get inputs
            inputs, targets = data
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            dicts = optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
                losses.append(current_loss/500)
                current_loss = 0.0

    import datetime
    # Print to indicate training completion
    print('Training process has finished.')
    # Current time as a string
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save the dictionaries containing updates, exp_avgs, params, grads
    dicts_filename = f"./results/dicts_{current_time}.pth"
    torch.save(dicts, dicts_filename)
    print(f"Saved parameter and gradient history to {dicts_filename}")

    # Save the losses
    losses_filename = f"./results/losses_{current_time}.pth"
    torch.save(losses, losses_filename)
    print(f"Saved losses to {losses_filename}")

