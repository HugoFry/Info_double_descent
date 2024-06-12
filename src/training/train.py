from torch import optim
import torch
import wandb

def train(model, train_dataloader, test_dataloader):
    """
    Need to think about the loss function:
        Should I add a head on the model to ouput only the last token?
        Don't I need to include a softmax over the output classes?
        I think I should use cross entropy loss.
    Need to return a list of train and test losses to be plotted.
    Also need to upload them live to WandB.
    """
    optimizer = optim.AdamW(model.parameters(), model.config.lr, model.config.weight_decay)
    
    for epoch in range(model.config.num_epochs):
        model.train()
        for batch_input, batch_labels in train_dataloader:
            optimizer.zero_grad()
            output = model(batch_input)
            train_loss = loss_fn(output, batch_labels)
            train_loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for batch_input, batch_labels in test_dataloader:
                output = model(batch_input)
                test_loss = loss_fn(output, batch_labels)
            
            #What else should I log here?
            wandb.log({
                'train_loss': train_loss.item(),
                'test_loss': test_loss.item(),
                }, step = epoch)