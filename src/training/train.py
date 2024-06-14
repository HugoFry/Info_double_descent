from torch import optim
from torch import nn
from dataclasses import asdict
import torch
import wandb

def loss_fn(output, labels):
    # Take the last token as the predicted probabilities
    probabilities = output[:,-1,:] 
    loss = nn.functional.cross_entropy(probabilities, labels)
    return loss
    

def train(model, train_dataloader, test_dataloader):
    """
    Need to include an lr scheduler - See Neel's notebook
    Also need to upload them live to WandB.
    """
    wandb.init(project = "Hugo_double_descent", config = asdict(model.config))
    
    optimizer = optim.AdamW(model.parameters(), model.config.lr, model.config.weight_decay)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(model.config.num_epochs):
        model.train()
        for batch_input, batch_labels in train_dataloader:
            optimizer.zero_grad()
            output = model(batch_input)
            train_loss = loss_fn(output, batch_labels)
            train_loss.backward()
            train_losses.append(train_loss.item())
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for batch_input, batch_labels in test_dataloader:
                output = model(batch_input)
                test_loss = loss_fn(output, batch_labels)
                test_losses.append(test_loss.item())
            
            #What else should I log here?
            wandb.log({
                'train_loss': train_loss.item(),
                'test_loss': test_loss.item(),
                }, step = epoch)
    return train_losses, test_losses