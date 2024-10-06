import sys
sys.path.append('/root/Info_double_descent/src')
import models
import datasets
from training import train

example_config = models.config.transformer_config(num_layers = 1,
                                                  frac_train = 0.4,
                                                  num_epochs = 500_000)
example_model = models.transformer.Transformer(example_config)
train_dataloader, test_dataloader = datasets.modular_addition.generate_test_train_split(example_model.config.prime, example_model.config.frac_train, example_model.config.seed)   
train.train(example_model, train_dataloader, test_dataloader)