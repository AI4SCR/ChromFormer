import torch
from ..processing.normalisation import centralize_and_normalize_torch
import torch.nn.functional as f
from .losses import biological_loss_fct, kabsch_loss_fct
import numpy as np
from .lddt_tools import loss_lddt
import math
from torch.nn import TransformerEncoderLayer
from typing import Tuple


class UniformLinear(torch.nn.Module):
    """Class that contains functions predict structures from given HiCs

    Args:
        nb_bins: number of loci per data point
        embeddin_size: 3D dimension
        batch_size: size of the batch
    """

    def __init__(self, nb_bins: int, embedding_size: int):
        self.nb_bins = nb_bins
        self.embedding_size = embedding_size
        super(UniformLinear, self).__init__()

        self.linear_encoder_layer_1 = torch.nn.Linear(self.nb_bins, 100)
        self.linear_encoder_layer_2 = torch.nn.Linear(100, 50)
        self.linear_encoder_layer_3 = torch.nn.Linear(50, self.embedding_size)

    def forward(
        self, x: torch.Tensor, is_training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward calculation to project HiCs using linear encoders with non linear activation inbetween each encoder layer

        Args:
            x: HiCs
            is_training: boolean that defines whether in training or testing mode

        Returns:
            z: 3D structures
            w: distance matrices
        """
        # x = torch.reshape(x, (self.batch_size, self.nb_bins, self.nb_bins))

        z = self.linear_encoder_layer_1(x)
        z = f.relu(z)
        z = self.linear_encoder_layer_2(z)
        z = f.relu(z)
        z = self.linear_encoder_layer_3(z)
        z = f.relu(z)
        z = centralize_and_normalize_torch(z)

        w = torch.cdist(z, z, p=2)

        return z, w


def train_uniform_linear(
    model,
    train_loader,
    train_dataset,
    optimizer,
    device,
    batch_size,
    nb_bins,
    embedding_size,
    lambda_bio,
    lambda_kabsch,
    distance_loss_fct,
):
    """training function for the linear model

    Args:
        model: the model to train
        train_loader: the data on which to train the model
        train_dataset: the dataset used for training
        optimizer: optimizer
        device: device to pass to the data for faster torch calculations
        batch_size: size of the batch
        nb_bins: number of loci per data point
        embedding_size: 3D dimension
        lambda_bio: biological loss weight
        lambda_kabsch: kabsch distance loss weight
        distance_loss_fct: Mean Square Error function

    Returns:
        The general training loss.
    """
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        pred_structure, pred_distance = model(data.hic_matrix)

        true_hic = data.hic_matrix.to(device)

        true_structure = data.structure_matrix.to(device)
        true_structure = torch.reshape(
            true_structure, (batch_size, nb_bins, embedding_size)
        )

        pred_distance = torch.reshape(pred_distance, (batch_size * nb_bins, nb_bins))
        true_distance = data.distance_matrix.to(device)

        # Biological loss
        biological_loss = biological_loss_fct(
            pred_structure,
            true_structure,
            pred_distance,
            true_distance,
            nb_bins,
            batch_size,
        )

        # Kabsch loss
        kabsch_loss = kabsch_loss_fct(
            pred_structure, true_structure, embedding_size, batch_size
        )

        # Distance loss
        distance_loss = distance_loss_fct(pred_distance, true_distance)

        # Combine losses
        loss = (
            lambda_bio * biological_loss + lambda_kabsch * kabsch_loss + distance_loss
        )

        loss.backward()

        loss_all += data.num_graphs * loss.item()

        optimizer.step()
    return loss_all / len(train_dataset)


def evaluate_uniform_linear(
    loader, model, device, batch_size, nb_bins, embedding_size, distance_loss_fct
):
    """Function to evaluate the Linear Model

    Args:
        loader: validation or training data
        model: the model to validate and test onr
        device: device to pass to the data for faster torch calculations
        batch_size: size of the batch
        nb_bins: number of loci per data point
        embedding_size: 3D dimension
        distance_loss_fct: Mean Square Error function

    Returns:
        Biological Loss value at this given epoch
        Kabsch distance loss value at this given epoch
        Distance loss value at this given epoch
        HiCs from the data
        Structures predicted by the mode
        True structures contained in the data
        Distance matrices predicted by the model
        True distance matrices contained in the data
    """
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

            pred_distance = torch.reshape(
                pred_distance, (batch_size * nb_bins, nb_bins)
            )

            true_hic = data.hic_matrix.detach().cpu()
            true_structure = data.structure_matrix.detach().cpu()
            true_distance = data.distance_matrix.detach().cpu()

            true_structure = torch.reshape(
                true_structure, (batch_size, nb_bins, embedding_size)
            )

            # Biological loss
            biological_loss = biological_loss_fct(
                pred_structure,
                true_structure,
                pred_distance,
                true_distance,
                nb_bins,
                batch_size,
            ).numpy()
            biological_losses.append(biological_loss)

            # Kabsch loss
            kabsch_loss = kabsch_loss_fct(
                pred_structure, true_structure, embedding_size, batch_size
            ).numpy()
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

    return (
        mean_biological_loss,
        mean_kabsch_loss,
        mean_distance_loss,
        true_hics,
        pred_structures,
        true_structures,
        pred_distances,
        true_distances,
    )


class ConfLinear(torch.nn.Module):
    """Class that contains functions predict structures from given HiCs

    Args:
        nb_bins: number of loci per data point
        embeddin_size: 3D dimension
        batch_size: size of the batch
        num_bins_logits: number of confidence bins to which predicted structures need to be projected
        zero_init: whether to use zero initialisation or the xavier initialisation
    """

    def __init__(
        self,
        nb_bins: int,
        embedding_size: int,
        num_bins_logits: int,
        zero_init: bool = True,
    ):
        self.nb_bins = nb_bins
        self.embedding_size = embedding_size
        self.zero_init = zero_init
        self.num_bins_logits = num_bins_logits
        super(ConfLinear, self).__init__()

        self.linear_encoder_layer_1 = torch.nn.Linear(self.nb_bins, 100)
        self.linear_encoder_layer_2 = torch.nn.Linear(100, 50)
        self.linear_encoder_layer_3 = torch.nn.Linear(50, self.embedding_size)
        self.linear_bin_layer = torch.nn.Linear(
            self.embedding_size, self.num_bins_logits
        )
        if self.zero_init:
            self.zero_initializer(self.linear_bin_layer)
        else:
            self.xavier_initializer(self.linear_bin_layer)  ## added

    def forward(self, x):
        """Forward calculation to project HiCs using linear encoders with non linear activation inbetween each encoder layer and then project structures in confidence bins

        Args:
            x: HiCs

        Returns:
            z: predicted 3D structures
            w: predicted distance matrices
            logits: predicted logits for belonging in each confidence bin
        """

        # x = torch.reshape(x, (self.batch_size, self.nb_bins, self.nb_bins))

        z = self.linear_encoder_layer_1(x)
        z = f.relu(z)
        z = self.linear_encoder_layer_2(z)
        z = f.relu(z)
        z = self.linear_encoder_layer_3(z)
        z = f.relu(z)
        z = centralize_and_normalize_torch(z)
        logits = f.relu(z)
        logits = self.linear_bin_layer(logits)  # added
        w = torch.cdist(z, z, p=2)

        return z, w, logits

    def zero_initializer(self, module):
        for name, param in module.named_parameters():
            if "weight" in name:
                torch.nn.init.constant_(param, 0.0)

    def xavier_initializer(self, module):

        for name, param in module.named_parameters():  # added
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.xavier_normal_(param)


def train_conf_linear(
    model,
    train_loader,
    train_dataset,
    optimizer,
    device,
    batch_size,
    nb_bins,
    embedding_size,
    lambda_bio,
    lambda_kabsch,
    distance_loss_fct,
    lambda_lddt,
    num_bins_logits,
):
    """training function for the linear model that calculates its own confidence

    Args:
        model: the model to train
        train_loader: the data on which to train the model
        train_dataset: the dataset used for training
        optimizer: optimizer
        device: device to pass to the data for faster torch calculations
        batch_size: size of the batch
        nb_bins: number of loci per data point
        embedding_size: 3D dimension
        lambda_bio: biological loss weight
        lambda_kabsch: kabsch distance loss weight
        distance_loss_fct: Mean Square Error function
        lambda_lddt: loss weight for the confidence learning
        num_bins_logits: number of confidence bins each loci was projected to

    Returns:
        The general training loss.
    """
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        pred_structure, pred_distance, logits = model(data.hic_matrix)
        # logits.retain_grad()
        true_hic = data.hic_matrix.to(device)

        true_structure = data.structure_matrix.to(device)
        true_structure = torch.reshape(
            true_structure, (batch_size, nb_bins, embedding_size)
        )

        pred_distance = torch.reshape(pred_distance, (batch_size * nb_bins, nb_bins))
        true_distance = data.distance_matrix.to(device)

        # Biological loss
        biological_loss = biological_loss_fct(
            pred_structure,
            true_structure,
            pred_distance,
            true_distance,
            nb_bins,
            batch_size,
        )

        # Kabsch loss
        kabsch_loss = kabsch_loss_fct(
            pred_structure, true_structure, embedding_size, batch_size
        )

        # Distance loss
        distance_loss = distance_loss_fct(pred_distance, true_distance)

        lddt_loss = loss_lddt(pred_structure, true_structure, logits, num_bins_logits)

        # Combine losses
        loss = (
            lambda_bio * biological_loss
            + lambda_kabsch * kabsch_loss
            + distance_loss
            + lambda_lddt * lddt_loss
        )
        loss.backward()
        loss_all += data.num_graphs * loss.item()

        optimizer.step()
    return loss_all / len(train_dataset)


def evaluate_conf_linear(
    loader,
    model,
    device,
    batch_size,
    nb_bins,
    embedding_size,
    distance_loss_fct,
    num_bins_logits,
):
    """Function to evaluate the Linear Model that computes confidence

    Args:
        loader: training or validation data
        model: the model to validate and test on
        device: device to pass to the data for faster torch calculations
        batch_size: size of the batch
        nb_bins: number of loci per data point
        embedding_size: 3D dimension
        distance_loss_fct: Mean Square Error function
        num_bins_logits: number of confidence bins each loci was projected to

    Returns:
        Biological Loss value at this given epoch
        Kabsch distance loss value at this given epoch
        Distance loss value at this given epoch
        HiCs from the data
        Structures predicted by the mode
        True structures contained in the data
        Distance matrices predicted by the model
        True distance matrices contained in the data
        Confidence loss value at this given epoch
    """
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

            pred_distance = torch.reshape(
                pred_distance, (batch_size * nb_bins, nb_bins)
            )

            true_hic = data.hic_matrix.detach().cpu()
            true_structure = data.structure_matrix.detach().cpu()
            true_distance = data.distance_matrix.detach().cpu()

            true_structure = torch.reshape(
                true_structure, (batch_size, nb_bins, embedding_size)
            )

            # Biological loss
            biological_loss = biological_loss_fct(
                pred_structure,
                true_structure,
                pred_distance,
                true_distance,
                nb_bins,
                batch_size,
            ).numpy()
            biological_losses.append(biological_loss)

            # Kabsch loss
            kabsch_loss = kabsch_loss_fct(
                pred_structure, true_structure, embedding_size, batch_size
            ).numpy()
            kabsch_losses.append(kabsch_loss)

            # Distance loss
            distance_loss = distance_loss_fct(pred_distance, true_distance).numpy()
            distance_losses.append(distance_loss)

            lddt_loss = loss_lddt(
                pred_structure, true_structure, logits, num_bins_logits
            )
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

    return (
        mean_biological_loss,
        mean_kabsch_loss,
        mean_distance_loss,
        true_hics,
        pred_structures,
        true_structures,
        pred_distances,
        true_distances,
        mean_lddt_loss,
    )


class TransformerModel(torch.nn.Module):
    """Class that contains functions for the transformer part of ChromFormer to predict structures from given HiCs. Modified version of PyTorch implementation example

    Args:
        n_token: 3D dimension for the decoder
        d_model: number of loci per data point
        nhead: number of transformer encoder heads
        d_hid: first embedding dimension and dimension of feedforward for the first encoder layer
        n_layers: number of layers (unused)
        dropout: dropout rate
        secd_hid: second feedfoward dimension used by the second transformer encoder layer
    """

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        secd_hid: int = 48,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_hid, dropout)
        self.encoder_layers = TransformerEncoderLayer(
            d_hid, nhead, dim_feedforward=d_hid, dropout=dropout
        )
        self.encoder_layers2 = TransformerEncoderLayer(
            d_hid, nhead, dim_feedforward=secd_hid, dropout=dropout
        )

        self.encoder = torch.nn.Linear(d_model, d_hid)
        self.d_model = d_model
        self.decoder = torch.nn.Linear(d_hid, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [seq_len,, seq_len, batch_size]

        Returns:
            outputpos Tensor of shape [seq_len, batch_size, ntoken]
            outputemb that contains the non decoded information
        """
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.encoder_layers(src)
        outputemb = self.encoder_layers2(output)
        outputpos = self.decoder(outputemb)
        return outputpos, outputemb


class PositionalEncoding(torch.nn.Module):
    """Positional embedding to encode. Modified version taken from pytorch example"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransConf(torch.nn.Module):
    """Class that contains ChromFormer to predict structures, distance matrices and loci level logit confidences from given HiCs

    Args:
        nb_bins: number of loci per data point
        embeddin_size: 3D dimension:
        batch_size: size of the batch
        num_bins_logits: number of confidence bins to which predicted structures need to be projected
        zero_init: whether to use zero initialisation or the xavier initialisation
        nhead: number of transformer encoder heads
        nb_hidden: first embedding dimension and dimension of feedforward for the first encoder layer
        n_layers: number of layers (unused)
        dropout: dropout rate
        secd_hid: second feedfoward dimension used by the second transformer encoder layer
    """

    def __init__(
        self, *,
        nb_bins: int,
        embedding_size: int,
        num_bins_logits: int,
        zero_init: bool = True,
        nb_head: int = 2,
        nb_hidden: int = 100,
        nb_layers: int = 1,
        dropout: float = 0.1,
        secd_hid: int = 48,
    ):
        self.nb_bins = nb_bins
        self.embedding_size = embedding_size
        self.zero_init = zero_init
        self.num_bins_logits = num_bins_logits
        self.nb_head = nb_head
        self.nb_hidden = nb_hidden
        self.nb_layers = nb_layers
        self.dropout = dropout
        self.secd_hid = secd_hid
        super(TransConf, self).__init__()

        self.transformer_encoder = TransformerModel(
            ntoken=self.embedding_size,
            d_model=self.nb_bins,
            nhead=self.nb_head,
            d_hid=self.nb_hidden,
            nlayers=self.nb_layers,
            dropout=self.dropout,
            secd_hid=self.secd_hid,
        )
        self.linear_bin_layer = torch.nn.Linear(self.nb_hidden, self.num_bins_logits)
        self.linear_bin_layer2 = torch.nn.Linear(
            self.num_bins_logits, self.num_bins_logits
        )
        if self.zero_init:
            self.zero_initializer(self.linear_bin_layer)
            self.zero_initializer(self.linear_bin_layer2)
        else:
            self.xavier_initializer(self.linear_bin_layer)
            self.xavier_initializer(self.linear_bin_layer2)

    def forward(self, x):
        """Uses the transformer model to find the embedding and 3D structures then uses 3D structures to derive distsance matrices and projects embedded information to derive confidence logits

        Args:
            x: HiC matrices

        Returns:
            z: predicted 3D structures
            w: predicted distance matrices
            logits: predicted logits for belonging in each confidence bin
        """

        x = x.permute(1, 0, 2)
        z, emb = self.transformer_encoder(x)
        z = z.permute(1, 0, 2)
        emb = emb.permute(1, 0, 2)

        z = centralize_and_normalize_torch(z)

        logits = self.linear_bin_layer(emb)
        logits = f.relu(logits)
        logits = self.linear_bin_layer2(logits)

        w = torch.cdist(z, z, p=2)

        return z, w, logits

    def zero_initializer(self, module):
        for name, param in module.named_parameters():
            if "weight" in name:
                torch.nn.init.constant_(param, 0.0)

    def xavier_initializer(self, module):

        for name, param in module.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.xavier_normal_(param)


def train_trans_conf(
    model,
    train_loader,
    train_dataset,
    optimizer,
    device,
    batch_size,
    nb_bins,
    embedding_size,
    lambda_bio,
    lambda_kabsch,
    distance_loss_fct,
    lambda_lddt,
    num_bins_logits,
):
    """training function for ChromFormer

    Args:
        model: the model to train
        train_loader: the data on which to train the model
        train_dataset: the dataset used for training
        optimizer: optimizer
        device: device to pass to the data for faster torch calculations
        batch_size: size of the batch
        nb_bins: number of loci per data point
        embedding_size: 3D dimension
        lambda_bio: biological loss weight
        lambda_kabsch: kabsch distance loss weight
        distance_loss_fct: Mean Square Error function
        lambda_lddt: loss weight for the confidence learning
        num_bins_logits: number of confidence bins each loci was projected to

    Returns:
        The general training loss.
    """
    model.train()

    loss_all = 0
    for data in train_loader:

        data = data.to(device)
        optimizer.zero_grad()

        pred_structure, pred_distance, logits = model(data.hic_matrix)
        true_hic = data.hic_matrix.to(device)
        true_structure = data.structure_matrix.to(device)
        true_structure = torch.reshape(
            true_structure, (batch_size, nb_bins, embedding_size)
        ).to(device)
        pred_distance = torch.reshape(
            pred_distance, (batch_size * nb_bins, nb_bins)
        ).to(device)
        true_distance = data.distance_matrix.to(device)

        # Biological loss
        biological_loss = biological_loss_fct(
            pred_structure,
            true_structure,
            pred_distance,
            true_distance,
            nb_bins,
            batch_size,
        )

        # Kabsch loss
        kabsch_loss = kabsch_loss_fct(
            pred_structure, true_structure, embedding_size, batch_size
        )

        # Distance loss
        distance_loss = distance_loss_fct(pred_distance, true_distance)

        lddt_loss = loss_lddt(pred_structure, true_structure, logits, num_bins_logits)

        # Combine losses

        loss = (
            lambda_bio * biological_loss
            + lambda_kabsch * kabsch_loss
            + distance_loss
            + lambda_lddt * lddt_loss
        )

        loss.backward()
        loss_all += data.num_graphs * loss.item()

        optimizer.step()
    return loss_all / len(train_dataset)


def evaluate_trans_conf(
    loader,
    model,
    device,
    batch_size,
    nb_bins,
    embedding_size,
    distance_loss_fct,
    num_bins_logits,
):
    """Function to evaluate ChromFormer

    Args:
        loader: training or validation data
        model: the model to validate and test on
        device: device to pass to the data for faster torch calculations
        batch_size: size of the batch
        nb_bins: number of loci per data point
        embedding_size: 3D dimension
        distance_loss_fct: Mean Square Error function
        num_bins_logits: number of confidence bins each loci was projected to

    Returns:
        Biological Loss value at this given epoch
        Kabsch distance loss value at this given epoch
        Distance loss value at this given epoch
        HiCs from the data
        Structures predicted by the mode
        True structures contained in the data
        Distance matrices predicted by the model
        True distance matrices contained in the data
        Confidence loss value at this given epoch
    """
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
            pred_structure = pred_structure.detach().to(device)
            pred_distance = pred_distance.detach().to(device)

            pred_distance = torch.reshape(
                pred_distance, (batch_size * nb_bins, nb_bins)
            )

            true_hic = data.hic_matrix.detach().to(device)
            true_structure = data.structure_matrix.detach().to(device)
            true_distance = data.distance_matrix.detach().to(device)

            true_structure = torch.reshape(
                true_structure, (batch_size, nb_bins, embedding_size)
            )

            # Biological loss
            biological_loss = (
                biological_loss_fct(
                    pred_structure,
                    true_structure,
                    pred_distance,
                    true_distance,
                    nb_bins,
                    batch_size,
                )
                .cpu()
                .numpy()
            )
            biological_losses.append(biological_loss)

            # Kabsch loss
            kabsch_loss = (
                kabsch_loss_fct(
                    pred_structure, true_structure, embedding_size, batch_size
                )
                .cpu()
                .numpy()
            )
            kabsch_losses.append(kabsch_loss)

            # Distance loss
            distance_loss = (
                distance_loss_fct(pred_distance, true_distance).cpu().numpy()
            )
            distance_losses.append(distance_loss)

            lddt_loss = (
                loss_lddt(pred_structure, true_structure, logits, num_bins_logits)
                .cpu()
                .numpy()
            )
            lddt_losses.append(lddt_loss)

            # To numpy
            true_hic = true_hic.cpu().numpy()

            pred_structure = pred_structure.cpu().numpy()
            true_structure = true_structure.cpu().numpy()

            pred_distance = pred_distance.cpu().numpy()
            true_distance = true_distance.cpu().numpy()

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

    return (
        mean_biological_loss,
        mean_kabsch_loss,
        mean_distance_loss,
        true_hics,
        pred_structures,
        true_structures,
        pred_distances,
        true_distances,
        mean_lddt_loss,
    )
