from torch.utils.data import Dataset, DataLoader
from ....models.transformer import Transformer

class ActivationsDataset(Dataset):
    def __init__(self, model: Transformer, model_train_dataloader, hook_name: str = ''):
        super().__init__()
        self.model = model
        self.model_train_dataloader = model_train_dataloader
        
        self.data
        
    def generate_dataset(self):
        for batch in self.model_train_dataloader:
            # Need to run the model with the correct hook. Pass the hook point into the __init__ method?
            # Need to specify the name of the desired module
            # Do the forward pass. Then index the dictionary with the desired name.
            self.model(batch['data']) # input has shape [batch_size, num_tokens] and the three tokens are [i, j, prime]
            pass
        pass
        
    def __getitem__(self, index: int):
        """
        Should return a dictionary of the form:
        {'inputs': Tensor([i,j], dtype = long), 'hidden embedding': Tensor(shape = [128])}
        """
        pass
    
    def __len__(self):
        pass