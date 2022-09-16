import torch
from ..Data_Tools.Normalisation import centralize_and_normalize_torch
import torch.nn.functional as f
from .losses import biological_loss_fct, kabsch_loss_fct
import numpy as np
from .lddt_tools import loss_lddt

class UniformLinear(torch.nn.Module):
    def __init__(self, nb_bins: int, embedding_size: int, batch_size: int):
        self.nb_bins = nb_bins
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        super(UniformLinear, self).__init__()
        
        self.linear_encoder_layer_1 = torch.nn.Linear(self.nb_bins, 100)
        self.linear_encoder_layer_2 = torch.nn.Linear(100, 50)
        self.linear_encoder_layer_3 = torch.nn.Linear(50, self.embedding_size)
        
    def forward(self, x, is_training=False):
        
        x = torch.reshape(x, (self.batch_size, self.nb_bins, self.nb_bins))
        
        z = self.linear_encoder_layer_1(x)
        z = f.relu(z)
        z = self.linear_encoder_layer_2(z)
        z = f.relu(z)
        z = self.linear_encoder_layer_3(z)
        z = f.relu(z)
        z = centralize_and_normalize_torch(z, self.embedding_size, self.nb_bins, self.batch_size)
        
        w = torch.cdist(z, z, p=2)
        
        return z, w

def train_uniform_linear(model, train_loader, train_dataset, optimizer, device, batch_size, nb_bins, embedding_size, lambda_bio, lambda_kabsch, distance_loss_fct):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        pred_structure, pred_distance = model(data.hic_matrix)
        
        true_hic = data.hic_matrix.to(device)
        
        true_structure = data.structure_matrix.to(device)
        true_structure = torch.reshape(true_structure, (batch_size, nb_bins, embedding_size))
        
        pred_distance = torch.reshape(pred_distance, (batch_size*nb_bins, nb_bins))
        true_distance = data.distance_matrix.to(device)
        
        # Biological loss
        biological_loss = biological_loss_fct(pred_structure, true_structure, pred_distance, true_distance, nb_bins, batch_size)

        # Kabsch loss
        kabsch_loss = kabsch_loss_fct(pred_structure, true_structure, embedding_size, batch_size)
        
        # Distance loss 
        distance_loss = distance_loss_fct(pred_distance, true_distance)
        
        # Combine losses
        loss = lambda_bio * biological_loss + lambda_kabsch * kabsch_loss + distance_loss
        
        loss.backward()
        
        loss_all += data.num_graphs * loss.item()
        
        optimizer.step()
    return loss_all / len(train_dataset)

def evaluate_uniform_linear(loader, model, device, batch_size, nb_bins, embedding_size, distance_loss_fct):
    model.eval()

    true_hics = []
    
    pred_structures = []
    true_structures = []
    
    pred_distances = []
    true_distances = []
    
    kabsch_losses = []
    distance_losses = []
    biological_losses = []

    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            
            pred_structure, pred_distance = model(data.hic_matrix)
            
            pred_structure = pred_structure.detach().cpu()
            pred_distance = pred_distance.detach().cpu()
            
            pred_distance = torch.reshape(pred_distance, (batch_size*nb_bins, nb_bins))
            
            true_hic = data.hic_matrix.detach().cpu()
            true_structure = data.structure_matrix.detach().cpu()
            true_distance = data.distance_matrix.detach().cpu()
            
            true_structure = torch.reshape(true_structure, (batch_size, nb_bins, embedding_size))
            
            # Biological loss
            biological_loss = \
                biological_loss_fct(pred_structure, true_structure, pred_distance, true_distance, nb_bins, batch_size).numpy()
            biological_losses.append(biological_loss)
            
            # Kabsch loss
            kabsch_loss = kabsch_loss_fct(pred_structure, true_structure, embedding_size, batch_size).numpy()
            kabsch_losses.append(kabsch_loss)
            
            # Distance loss
            distance_loss = distance_loss_fct(pred_distance, true_distance).numpy()
            distance_losses.append(distance_loss)
            
            # To numpy
            true_hic = true_hic.numpy()
            
            pred_structure = pred_structure.numpy()
            true_structure = true_structure.numpy()
            
            pred_distance = pred_distance.numpy()
            true_distance = true_distance.numpy()
            
            # Store results
            true_hics.append(true_hic)
            
            pred_structures.append(pred_structure)
            true_structures.append(true_structure)
            
            pred_distances.append(pred_distance)
            true_distances.append(true_distance)
    
    # Format results
    true_hics = np.vstack(true_hics)
    
    pred_structures = np.vstack(pred_structures)
    true_structures = np.vstack(true_structures)
    
    pred_distances = np.vstack(pred_distances)
    true_distances = np.vstack(true_distances)
    
    # Compute mean losses
    mean_biological_loss = np.mean(np.asarray(biological_loss).flatten())
    mean_kabsch_loss = np.mean(np.asarray(kabsch_losses).flatten())
    mean_distance_loss = np.mean(np.asarray(distance_losses).flatten())
    
    return mean_biological_loss, mean_kabsch_loss, mean_distance_loss, true_hics, \
            pred_structures, true_structures, pred_distances, true_distances

class ConfLinear(torch.nn.Module):
    def __init__(self, nb_bins: int, embedding_size: int, batch_size: int, num_bins_logits: int, zero_init: bool = True):
        self.nb_bins = nb_bins
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.zero_init = zero_init
        super(ConfLinear, self).__init__()
        
        self.linear_encoder_layer_1 = torch.nn.Linear(self.nb_bins, 100)
        self.linear_encoder_layer_2 = torch.nn.Linear(100, 50)
        self.linear_encoder_layer_3 = torch.nn.Linear(50, self.embedding_size)
        self.linear_bin_layer = torch.nn.Linear(self.embedding_size, num_bins_logits)
        if self.zero_init:
            self.zero_initializer(self.linear_bin_layer)
        else:
            self.xavier_initializer(self.linear_bin_layer) ## added
    def forward(self, x):
        
        x = torch.reshape(x, (self.batch_size, self.nb_bins, self.nb_bins))
        
        z = self.linear_encoder_layer_1(x)
        z = f.relu(z)
        z = self.linear_encoder_layer_2(z)
        z = f.relu(z)
        z = self.linear_encoder_layer_3(z)
        z = f.relu(z)
        z = centralize_and_normalize_torch(z, self.embedding_size, self.nb_bins, self.batch_size)
        logits = f.relu(z)
        logits = self.linear_bin_layer(logits) # added
        w = torch.cdist(z, z, p=2)
        
        return z, w, logits
    def zero_initializer(self, module):
        for name, param in module.named_parameters():
            if 'weight' in name:
                torch.nn.init.constant_(param, 0.0)
    def xavier_initializer(self, module):
        
        for name, param in module.named_parameters(): # added
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_normal_(param)
    
def train_conf_linear(model, train_loader, train_dataset, optimizer, device, batch_size, nb_bins, embedding_size, lambda_bio, lambda_kabsch, distance_loss_fct, lambda_lddt, num_bins_logits):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        pred_structure, pred_distance, logits = model(data.hic_matrix)
        #logits.retain_grad()
        true_hic = data.hic_matrix.to(device)
        
        true_structure = data.structure_matrix.to(device)
        true_structure = torch.reshape(true_structure, (batch_size, nb_bins, embedding_size))
        
        pred_distance = torch.reshape(pred_distance, (batch_size*nb_bins, nb_bins))
        true_distance = data.distance_matrix.to(device)
        
        # Biological loss
        biological_loss = biological_loss_fct(pred_structure, true_structure, pred_distance, true_distance, nb_bins, batch_size)
        
        # Kabsch loss
        kabsch_loss = kabsch_loss_fct(pred_structure, true_structure, embedding_size, batch_size)
        
        # Distance loss 
        distance_loss = distance_loss_fct(pred_distance, true_distance)
        
        lddt_loss = loss_lddt(pred_structure, true_structure, logits, num_bins_logits)

        
        # Combine losses
        loss = lambda_bio * biological_loss + lambda_kabsch * kabsch_loss + distance_loss + lambda_lddt*lddt_loss
        loss.backward()
        loss_all += data.num_graphs * loss.item()
                
        optimizer.step()
    return loss_all / len(train_dataset)

def evaluate_conf_linear(loader, model, device, batch_size, nb_bins, embedding_size, distance_loss_fct, num_bins_logits):
    model.eval()

    true_hics = []
    
    pred_structures = []
    true_structures = []
    
    pred_distances = []
    true_distances = []
    
    kabsch_losses = []
    distance_losses = []
    biological_losses = []
    lddt_losses = []

    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            
            pred_structure, pred_distance, logits = model(data.hic_matrix)
            pred_structure = pred_structure.detach().cpu()
            pred_distance = pred_distance.detach().cpu()
            
            pred_distance = torch.reshape(pred_distance, (batch_size*nb_bins, nb_bins))
            
            true_hic = data.hic_matrix.detach().cpu()
            true_structure = data.structure_matrix.detach().cpu()
            true_distance = data.distance_matrix.detach().cpu()
            
            true_structure = torch.reshape(true_structure, (batch_size, nb_bins, embedding_size))
            
            # Biological loss
            biological_loss = \
                biological_loss_fct(pred_structure, true_structure, pred_distance, true_distance, nb_bins, batch_size).numpy()
            biological_losses.append(biological_loss)
            
            # Kabsch loss
            kabsch_loss = kabsch_loss_fct(pred_structure, true_structure, embedding_size, batch_size).numpy()
            kabsch_losses.append(kabsch_loss)
            
            # Distance loss
            distance_loss = distance_loss_fct(pred_distance, true_distance).numpy()
            distance_losses.append(distance_loss)
            
            lddt_loss = loss_lddt(pred_structure, true_structure, logits, num_bins_logits)
            lddt_losses.append(lddt_loss)

            
            # To numpy
            true_hic = true_hic.numpy()
            
            pred_structure = pred_structure.numpy()
            true_structure = true_structure.numpy()
            
            pred_distance = pred_distance.numpy()
            true_distance = true_distance.numpy()
            
            # Store results
            true_hics.append(true_hic)
            
            pred_structures.append(pred_structure)
            true_structures.append(true_structure)
            
            pred_distances.append(pred_distance)
            true_distances.append(true_distance)
    
    # Format results
    true_hics = np.vstack(true_hics)
    
    pred_structures = np.vstack(pred_structures)
    true_structures = np.vstack(true_structures)
    
    pred_distances = np.vstack(pred_distances)
    true_distances = np.vstack(true_distances)
    
    # Compute mean losses
    mean_biological_loss = np.mean(np.asarray(biological_loss).flatten())
    mean_kabsch_loss = np.mean(np.asarray(kabsch_losses).flatten())
    mean_distance_loss = np.mean(np.asarray(distance_losses).flatten())
    mean_lddt_loss = np.mean(np.asarray(lddt_losses).flatten())

    
    return mean_biological_loss, mean_kabsch_loss, mean_distance_loss, true_hics, \
            pred_structures, true_structures, pred_distances, true_distances, mean_lddt_loss