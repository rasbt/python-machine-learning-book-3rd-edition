# coding: utf-8


import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
from torchvision.datasets import MNIST 
from torchvision import transforms
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, DiskSaver
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

# # Chapter 13: Going Deeper -- the Mechanics of PyTorch (Part 3/3)

# ## Higher-level PyTorch APIs: a short introduction to PyTorch-Ignite 

# ### Setting up the PyTorch model



 
 
 
image_path = './' 
torch.manual_seed(1) 
 
transform = transforms.Compose([ 
    transforms.ToTensor() 
]) 
 
 
mnist_train_dataset = MNIST( 
    root=image_path,  
    train=True,
    transform=transform,  
    download=True
) 
 
mnist_val_dataset = MNIST( 
    root=image_path,  
    train=False,  
    transform=transform,  
    download=False 
) 
 
batch_size = 64
train_loader = DataLoader( 
    mnist_train_dataset, batch_size, shuffle=True 
) 
 
val_loader = DataLoader( 
    mnist_val_dataset, batch_size, shuffle=False 
) 
 
 
def get_model(image_shape=(1, 28, 28), hidden_units=(32, 16)): 
    input_size = image_shape[0] * image_shape[1] * image_shape[2] 
    all_layers = [nn.Flatten()]
    for hidden_unit in hidden_units: 
        layer = nn.Linear(input_size, hidden_unit) 
        all_layers.append(layer) 
        all_layers.append(nn.ReLU()) 
        input_size = hidden_unit 
 
    all_layers.append(nn.Linear(hidden_units[-1], 10)) 
    all_layers.append(nn.Softmax(dim=1)) 
    model = nn.Sequential(*all_layers)
    return model 
 
 
device = "cuda" if torch.cuda.is_available() else "cpu"
 
model = get_model().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ### Setting up training and validation engines with PyTorch-Ignite



 
 
trainer = create_supervised_trainer(
    model, optimizer, loss_fn, device=device
)
 
val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(loss_fn)
}
 
evaluator = create_supervised_evaluator(
    model, metrics=val_metrics, device=device
)


# ### Creating event handlers for logging and validation



# How many batches to wait before logging training status
log_interval = 100
 
@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss():
    e = trainer.state.epoch
    max_e = trainer.state.max_epochs
    i = trainer.state.iteration
    batch_loss = trainer.state.output
    print(f"Epoch[{e}/{max_e}], Iter[{i}] Loss: {batch_loss:.2f}")




@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results():
    eval_state = evaluator.run(val_loader)
    metrics = eval_state.metrics
    e = trainer.state.epoch
    max_e = trainer.state.max_epochs
    acc = metrics['accuracy']
    avg_loss = metrics['loss']
    print(f"Validation Results - Epoch[{e}/{max_e}] Avg Accuracy: {acc:.2f} Avg Loss: {avg_loss:.2f}")


# ### Setting up training checkpoints and saving the best model



 
# We will save in the checkpoint the following:
to_save = {"model": model, "optimizer": optimizer, "trainer": trainer}
 
# We will save checkpoints to the local disk
output_path = "./output"
save_handler = DiskSaver(dirname=output_path, require_empty=False)
 
# Set up the handler:
checkpoint_handler = Checkpoint(
    to_save, save_handler, filename_prefix="training")

# Attach the handler to the trainer
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)




# Store best model by validation accuracy
best_model_handler = Checkpoint(
    {"model": model},
    save_handler,
    filename_prefix="best",
    n_saved=1,
    score_name="accuracy",
    score_function=Checkpoint.get_default_score_fn("accuracy"),
)
 
evaluator.add_event_handler(Events.COMPLETED, best_model_handler)


# ### Setting up TensorBoard as an experiment tracking system



 
 
tb_logger = TensorboardLogger(log_dir=output_path)
 
# Attach handler to plot trainer's loss every 100 iterations
tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)
 
# Attach handler for plotting both evaluators' metrics after every epoch completes
tb_logger.attach_output_handler(
    evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="validation",
    metric_names="all",
    global_step_transform=global_step_from_engine(trainer),
)


# ### Executing the PyTorch-Ignite model training code



trainer.run(train_loader, max_epochs=5)


# ---
# 
# Readers may ignore the next cell.




