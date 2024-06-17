from torch import optim
from torch import nn
from dataclasses import asdict
from tqdm import trange
from math import sqrt
import torch
import wandb

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

def get_accuracy(output, labels):
    # Take the last token as the predicted logits
    logits = output[:,-1,:]
    probabilities = nn.functional.softmax(logits.to(torch.float64), dim = -1)
    accuracy = torch.gather(probabilities, -1, labels.unsqueeze(-1)).squeeze().mean().item()
    return accuracy
    

def train(model, train_dataloader, test_dataloader):
    """
    To do:
        Upload activation norms:
            Residual in various locations (l2)
            Hidden MLP (l0, l1, l2)
            
        Upload gradients to WnB:
            l2 norm gradients
            max gradients
        
    Questions I want to understand:
        Why do we get the shape of the loss curves around the spikes? eg why do we get a mini double descent in the train loss
        How does weight decay fit into this?
        Has this got anything to do with the curvature in the loss landscape?
    """
    wandb.init(project = "Hugo_double_descent", config = asdict(model.config))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using the following device: {device}!')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr = model.config.lr, weight_decay = model.config.weight_decay, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in trange(model.config.num_epochs):
        epoch_train_loss = 0
        epoch_max_train_loss = 0
        epoch_train_accuracy = 0
        epoch_test_loss = 0
        epoch_test_accuracy = 0
        
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
        
        #Break this up into functions. It's a mess at the moment.
        model.eval()
        with torch.no_grad():
            for batch_data in train_dataloader:
                batch_input = batch_data['data'].to(device)
                batch_labels = batch_data['label'].to(device)
                output = model(batch_input)
                train_loss = loss_fn(output, batch_labels)
                max_train_loss = get_max_loss_in_batch(output, batch_labels)
                epoch_max_train_loss += max_train_loss
                epoch_train_loss += train_loss.item()
                train_accuracy = get_accuracy(output, batch_labels)
                epoch_train_accuracy += train_accuracy
            epoch_train_loss /= len(train_dataloader)
            epoch_train_accuracy /= len(train_dataloader)
            epoch_max_train_loss /= len(train_dataloader)
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_accuracy)
            
            for batch_data in test_dataloader:
                batch_input = batch_data['data'].to(device)
                batch_labels = batch_data['label'].to(device)
                output = model(batch_input)
                test_loss = loss_fn(output, batch_labels)
                epoch_test_loss += test_loss.item()
                test_accuracy = get_accuracy(output, batch_labels)
                epoch_test_accuracy += test_accuracy
            epoch_test_loss /= len(test_dataloader)
            epoch_test_accuracy /= len(train_dataloader)
            test_losses.append(epoch_test_loss)
            test_accuracies.append(epoch_test_accuracy)
            
            if model.config.wandb:
                wandb.log({
                    'train_loss': epoch_train_loss,
                    'max_train_loss': epoch_max_train_loss,
                    'train_accuracy': epoch_train_accuracy,
                    'test_loss': epoch_test_loss,
                    'test_accuracy': epoch_test_accuracy
                    }, step = epoch)
                
                if model.config.log_weight_norms:
                    mlp_encoder_weights_norm = torch.norm(model.layers[-1].mlp.encoder.weight.data.clone().detach()).item()
                    mlp_decoder_weights_norm = torch.norm(model.layers[-1].mlp.decoder.weight.data.clone().detach()).item()
                    mlp_total_weight_norm = sqrt(mlp_encoder_weights_norm**2 + mlp_decoder_weights_norm**2)
                    attention_q_weights_norm = torch.norm(model.layers[-1].attention.Q.clone().detach()).item()
                    attention_k_weights_norm = torch.norm(model.layers[-1].attention.K.clone().detach()).item()
                    attention_v_weights_norm = torch.norm(model.layers[-1].attention.V.clone().detach()).item()
                    attention_o_weights_norm = torch.norm(model.layers[-1].attention.O.clone().detach()).item()
                    attention_total_weight_norm = sqrt(attention_q_weights_norm**2 + attention_k_weights_norm**2 + attention_v_weights_norm**2 + attention_o_weights_norm**2)
                    layer_total_weight_norm = sqrt(mlp_total_weight_norm**2 + attention_total_weight_norm**2)
                    embedding_weights_norm = torch.norm(model.embed.embedding.weight.data.clone().detach()).item()
                    unembedding_weights_norm = torch.norm(model.unembed.unembedding.weight.data.clone().detach()).item()
                    positional_embedding_weights_norm = torch.norm(model.pos_embed.positional_embedding.clone().detach()).item()
                    
                    wandb.log({
                        'weight_norms/mlp_in': mlp_encoder_weights_norm,
                        'weight_norms/mlp_out': mlp_decoder_weights_norm,
                        'weight_norms/mlp_total': mlp_total_weight_norm,
                        'weight_norms/attention_q': attention_q_weights_norm,
                        'weight_norms/attention_k': attention_k_weights_norm,
                        'weight_norms/attention_v': attention_v_weights_norm,
                        'weight_norms/attention_o': attention_o_weights_norm,
                        'weight_norms/attention_total': attention_total_weight_norm,
                        'weight_norms/layer_total': layer_total_weight_norm,
                        'weight_norms/embedding': embedding_weights_norm,
                        'weight_norms/unembedding': unembedding_weights_norm,
                        'weight_norms/positional_embedding': positional_embedding_weights_norm,
                    }, step = epoch)
                    
                if model.config.log_activation_norms:
                    pass
                
                if model.config.log_gradient_norms:
                    pass
                
    return train_losses, train_accuracies, test_losses, test_accuracies