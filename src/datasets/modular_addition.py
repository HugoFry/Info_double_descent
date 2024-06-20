from torch.utils.data import Dataset, DataLoader, random_split
import torch

class modular_addition(Dataset):
    def __init__(self, prime: int):
        super().__init__()
        self.prime = prime
        self.data = [torch.tensor([i, j, prime]) for i in range(prime) for j in range(prime)]
        self.labels = [torch.tensor((i + j) % prime, dtype=torch.long) for i in range(prime) for j in range(prime)]
        
    def __len__(self):
        return self.prime**2
        
    def __getitem__(self, index):
        return {'data': self.data[index], 'label': self.labels[index]}
    
def generate_test_train_split(prime: int, proportion_train: float, seed: int,):
    torch.manual_seed(seed)
    dataset = modular_addition(prime)
    train_length = int(len(dataset)*proportion_train)
    test_length = len(dataset) - train_length
    train_dataset, test_dataset = random_split(dataset, [train_length, test_length])
    
    # Create data loader, using full batch training so don't need to shuffle the data.
    train_dataloader = DataLoader(train_dataset, batch_size = len(train_dataset), shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = len(test_dataset), shuffle = False)
    return train_dataloader, test_dataloader