from torch import optim
from torch import nn
from dataclasses import asdict
from tqdm import trange
from math import sqrt
import torch
import wandb
import os
import re

def loss_fn(output, labels):
    # Take the last token as the predicted logits
    logits = output[:,-1,:]
    loss = nn.functional.cross_entropy(logits.to(torch.float64), labels)
    return loss

def get_max_loss_in_batch(output, labels):
    # Take the last token as the predicted logits
    logits = output[:,-1,:]
    loss = nn.functional.cross_entropy(logits.to(torch.float64), labels, reduction = 'none')
    # Could alos return variance in the future?
    max_loss = loss.max().item()
    return max_loss

def get_full_loss(output, labels):
    # Take the last token as the predicted logits
    logits = output[:,-1,:]
    loss = nn.functional.cross_entropy(logits.to(torch.float64), labels, reduction = 'none')
    # Could alos return variance in the future?
    return loss

def get_accuracy(output, labels):
    # Take the last token as the predicted logits
    logits = output[:,-1,:]
    probabilities = nn.functional.softmax(logits.to(torch.float64), dim = -1)
    accuracy = torch.gather(probabilities, -1, labels.unsqueeze(-1)).squeeze().mean().item()
    return accuracy

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir = '../src/checkpoints'):
    checkpoint_dict = {
        'config': model.config,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    path = f'{checkpoint_dir}/epoch_{epoch}'
    torch.save(checkpoint_dict, path)
    
def delete_old_checkpoints(current_epoch, num_checkpoints = 20, checkpoint_dir = '../src/checkpoints'):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir)]
    epoch_pattern = re.compile(r'epoch_(\d+)')
    for file in checkpoint_files:
        match = epoch_pattern.search(file)
        if match:
            epoch = int(match.group(1))
            if epoch <= current_epoch - num_checkpoints:
                os.remove(f'{checkpoint_dir}/{file}')

def train(model, train_dataloader, test_dataloader):
    """
    To do:
        Need to think about whether I should mean over token position etc.
        Need to clarify whether backward hooks are acutal gradients or optimizer time weighted gradients
        
        Upload gradients to WnB:
                l2 norm gradients
                max gradients
                
        Upload activation norms:
            Residual in various locations (l2)
            Hidden MLP (l0, l1, l2)
        
    Questions I want to understand:
        Why do we get the shape of the loss curves around the spikes? eg why do we get a mini double descent in the train loss (Does the gradient give this value...?)
        How does weight decay fit into this?
        Has this got anything to do with the curvature in the loss landscape?
        The weight norms suggest that the model is constantly fighting against weight decay. How does this play out mathematically?
            It is qualitatively different to l2 loss: the model will update in the radial direction when using AdamW...
    """
    assert len(train_dataloader) == 1 and len(test_dataloader) == 1, "All my evals are only done assuming full batch training"
    
    if model.config.wandb:
        wandb.init(project = "Hugo_double_descent", config = asdict(model.config))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using the following device: {device}!')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr = model.config.lr, weight_decay = model.config.weight_decay, betas=model.config.betas)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))
    
    for epoch in trange(model.config.num_epochs):
        model.train()
        for batch_data in train_dataloader:
            batch_input = batch_data['data'].to(device)
            batch_labels = batch_data['label'].to(device)
            optimizer.zero_grad()
            output = model(batch_input)
            train_loss = loss_fn(output, batch_labels)
            train_loss.backward()
            optimizer.step()
            scheduler.step()
        
        if model.config.wandb:
            model.eval()
            train_loss = evaluate_on_train(model, train_dataloader, optimizer, epoch, device)
            if train_loss < 1e-4:
                model.config.has_trained = True
            evaluate_on_test(model, test_dataloader, epoch, device)
            if model.config.log_weight_norms:
                log_wandb_model_weight_norms(model, epoch)
            if model.config.log_optimizer_moments_norm:
                log_optimizer_moments_norm(model, optimizer, scheduler, epoch)
            if model.config.save_checkpoints:
                delete_old_checkpoints(epoch)
                save_checkpoint(model, optimizer, epoch, train_loss)
                if model.config.has_trained and train_loss > 1e-4:
                    model.config.save_checkpoints = False
    wandb.finish()


def log_optimizer_moments_norm(model, optimizer, scheduler, epoch):
    optimizer_moments = {}
    for name, param in model.named_parameters():
        state = optimizer.state[param]
        m = state['exp_avg'].clone()
        v = state['exp_avg_sq'].clone()
        delta = torch.norm(scheduler.get_last_lr()[0] * m/(torch.sqrt(v) + 1e-8))
        m = torch.norm(m)
        v = torch.norm(v)
        optimizer_moments[f'moments/m/{name}'] = m
        optimizer_moments[f'moments/v/{name}'] = v
        optimizer_moments[f'moments/delta/{name}'] = delta
    wandb.log(optimizer_moments, step = epoch)


def evaluate_on_train(model, train_dataloader, optimizer, epoch, device):    
    for batch_data in train_dataloader:
        batch_input = batch_data['data'].to(device)
        batch_labels = batch_data['label'].to(device)
                
        with model.evaluation_hooks():
            if model.config.log_gradient_norms:
                optimizer.zero_grad()
                output = model(batch_input)
                train_loss = loss_fn(output, batch_labels)
                train_loss.backward()
                log_wandb_model_gradient_norms(model, epoch)
            else:
                with torch.no_grad():
                    output = model(batch_input)
                    train_loss = loss_fn(output, batch_labels)
            cache = model.cache
        
        max_train_loss = get_max_loss_in_batch(output, batch_labels)
        train_accuracy = get_accuracy(output, batch_labels)
    
    '''
    Note that the training dynamics of AdamW and Adam are VERY different.
    The wight decay term is the dominant term for total loss, and yet contributes a similar amount to the parameter updates
    due to the adaptive component of Adam. Strange. I need to think about it much more...
    '''
    l2_loss = l2_reg(model)
    
    wandb.log({
            'train_loss': train_loss.item(),
            'l2_loss': l2_loss,
            'max_train_loss': max_train_loss,
            'train_accuracy': train_accuracy,
            }, step = epoch)
    
    if cache:
        assert model.config.log_activation_norms, "Hooks have been incorrectly registered"
        for key, value in cache.items():
            assert '_grad' not in key, "Backward hooks are not yet supported"
            cache[key] = torch.norm(value, dim = -1).mean().item()
        wandb.log({f'activations/{key}': value for key, value in cache.items()}, step = epoch)
    
    return train_loss.item()
    
def evaluate_on_test(model, test_dataloader, epoch, device):
    for batch_data in test_dataloader:
        batch_input = batch_data['data'].to(device)
        batch_labels = batch_data['label'].to(device)
        with torch.no_grad():
            output = model(batch_input)
        test_loss = loss_fn(output, batch_labels)
        test_accuracy = get_accuracy(output, batch_labels)
    
    if model.config.wandb:
        wandb.log({
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
            }, step = epoch)
        

def log_wandb_model_weight_norms(model, epoch):
    weight_norms_dict = {}
    total = 0
    for name, parameter in model.named_parameters():
        norm = torch.norm(parameter.clone().detach()).item()
        weight_norms_dict[f'weight_norms/{name}'] = norm
        total += norm**2
    total = sqrt(total)
    weight_norms_dict['weight_norms/total'] = total
    wandb.log(weight_norms_dict, step = epoch)
    
def l2_reg(model):
    total = 0
    for name, parameter in model.named_parameters():
        norm = torch.norm(parameter.clone().detach()).item()
        total += norm**2
    return total
        
def log_wandb_model_gradient_norms(model, epoch):
    gradient_norms_dict = {}
    for name, parameter in model.named_parameters():
        gradient_norms_dict[f'gradients/{name}'] = torch.norm(parameter.grad.clone().detach()).item()
    wandb.log(gradient_norms_dict, step = epoch)