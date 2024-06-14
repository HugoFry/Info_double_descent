from torch import optim
from torch import nn
from dataclasses import asdict
import torch
import wandb

def loss_fn(output, labels):
    # Take the last token as the predicted logits
    logits = output[:,-1,:]
    loss = nn.functional.cross_entropy(logits.to(torch.float64), labels)
    return loss
    

def train(model, train_dataloader, test_dataloader):
    """
    Need to include an lr scheduler - See Neel's notebook
    Also need to upload them live to WandB.
    """
    wandb.init(project = "Hugo_double_descent", config = asdict(model.config))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using the following device: {device}!')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr = model.config.lr, weight_decay = model.config.weight_decay, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))
    
    train_losses = []
    test_losses = []
    
    for epoch in range(model.config.num_epochs):
        epoch_train_loss = 0
        epoch_test_loss = 0
        
        model.train()
        for batch_data in train_dataloader:
            batch_input = batch_data['data'].to(device)
            batch_labels = batch_data['label'].to(device)
            optimizer.zero_grad()
            output = model(batch_input)
            train_loss = loss_fn(output, batch_labels)
            train_loss.backward()
            epoch_train_loss += train_loss.item()            
            optimizer.step()
            scheduler.step()
        epoch_train_loss /= len(train_dataloader)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        with torch.no_grad():
            for batch_data in test_dataloader:
                batch_input = batch_data['data'].to(device)
                batch_labels = batch_data['label'].to(device)
                output = model(batch_input)
                test_loss = loss_fn(output, batch_labels)
                epoch_test_loss += test_loss.item()
            epoch_test_loss /= len(test_dataloader)
            test_losses.append(epoch_test_loss)
            
            #What else should I log here?
            wandb.log({
                'train_loss': epoch_train_loss,
                'test_loss': epoch_test_loss,
                }, step = epoch)
    return train_losses, test_losses