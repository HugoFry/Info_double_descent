"""
Need to specify the MINE loss function
"""
import torch
from torch.utils.data import DataLoader
from ..models.MINE import MINENetwork

def loss_function(self, joint_output, marginal_output):
    """
    Loss function given in the MINE paper.
    The form of the loss funciton is given by the Donsker-Varadhan representation.
    """
    joint_loss = torch.mean(joint_output, dim = 0)
    marginal_loss = torch.log(torch.mean(torch.exp(marginal_output), dim = 0))
    loss = joint_loss - marginal_loss
    return loss


def train_MINE(GNN_model, GNN_dataset, MINE_models: MINENetwork, device):
    """
    To do:
        Think about a test/train split
        Think about logging to WnB. What should I log?
    """
    MINE_dataset = activations_dataset(GNN_model, GNN_dataset, MINE_models.config.activations_dataset_size)
    MINE_iterable_dataloader = iter(DataLoader(MINE_dataset, batch_size = MINE_models.config.batch_size, shuffle = True))
    
    for layer in range(len(MINE_models)):
        for network_type in ['input', 'label']:
            MINE_models[network_type][layer].train()
    
    epoch = 0
    while epoch < MINE_models.config.epochs:
        #Draw two batches of data from the dataset to create the marginal and joint distributions
        try:
            batch_a = next(MINE_iterable_dataloader)
            batch_b = next(MINE_iterable_dataloader)
        except StopIteration:
            epoch +=1
            MINE_iterable_dataloader = iter(DataLoader(MINE_dataset, batch_size = MINE_models.config.batch_size, shuffle = True))
            batch_a = next(MINE_iterable_dataloader)
            batch_b = next(MINE_iterable_dataloader)
        
        joint_data = {
            'input': [torch.cat((batch_a[0], batch_a[layer+1]), dim = -1) for layer in len(batch_a)-2],
            'label': [torch.cat((batch_a[-1], batch_a[layer+1]), dim = -1) for layer in len(batch_a)-2],
            }
        
        marginal_data = {
            'input': [torch.cat((batch_a[0], batch_b[layer+1]), dim = -1) for layer in len(batch_a)-2],
            'label': [torch.cat((batch_a[-1], batch_b[layer+1]), dim = -1) for layer in len(batch_a)-2],
            }
        
        for layer in range(len(MINE_models)):
            for network_type in ['input', 'label']:
                model = MINE_models[network_type][layer]
                model.optimizer.zero_grad()
                joint_output = model(joint_data[network_type][layer])
                marginal_output = model(marginal_data[network_type][layer])
                loss = MINE_models.loss_function(joint_output, marginal_output)
                loss.backward()
                model.optimizer.step()