import pytorch_lightning as pylight
import torch
from .losses import biological_loss_fct, kabsch_loss_fct, kabsch_distance_numpy
from .lddt_tools import loss_lddt
import numpy as np

class LitChromFormer(pylight.LightningModule):
    def __init__(self,
                 model,
                 optimizer,
                 optimizer_kwargs,
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
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.nb_bins = nb_bins
        self.embedding_size = embedding_size
        self.lambda_bio = lambda_bio
        self.lambda_kabsch = lambda_kabsch
        self.distance_loss_fct = distance_loss_fct
        self.lambda_lddt = lambda_lddt
        self.num_bins_logits = num_bins_logits

        self.model = model
        self.save_hyperparameters(ignore=['model', 'distance_loss_fct'])

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        self.model.train()
        batch_size = len(batch[0])

        # batch[0] := hic matrix
        # batch[1] := structure
        # batch[2] := distance
        true_hic, true_structure, true_distance = batch
        pred_structure, pred_distance, logits = self.model(true_hic)

        assert pred_structure.shape == true_structure.shape
        assert pred_distance.shape == true_distance.shape

        # TODO: Why reshape?
        _pred_distance = pred_distance.reshape((batch_size * self.nb_bins, self.nb_bins))

        # Biological loss
        biological_loss = biological_loss_fct(
            pred_structure,
            true_structure,
            pred_distance,
            true_distance,
            self.nb_bins,
            batch_size,
        )

        # Kabsch loss
        kabsch_loss = kabsch_loss_fct(
            pred_structure, true_structure, self.embedding_size, batch_size
        )

        # TODO: fix input
        kabsch_distances = []
        for true_structure in true_structure:
            kabsch_distances.append(
                kabsch_distance_numpy(
                    pred_structure, true_structure, self.embedding_size
                )
            )
        kabsch_distance = np.mean(kabsch_distances)

        # Distance loss
        distance_loss = self.distance_loss_fct(pred_distance, true_distance)

        lddt_loss = loss_lddt(pred_structure, true_structure, logits, self.num_bins_logits)

        # Combine losses
        loss = (
                self.lambda_bio * biological_loss
                + self.lambda_kabsch * kabsch_loss
                + distance_loss
                + self.lambda_lddt * lddt_loss
        )
        # TODO: Do I need to call this here?
        loss.backward()
        loss = batch.num_graphs * loss.item()

        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        batch_size = len(batch)

        pred_structure, pred_distance, logits = self.model(batch.hic_matrix)
        pred_structure = pred_structure.detach().cpu()
        pred_distance = pred_distance.detach().cpu()

        pred_distance = torch.reshape(
            pred_distance, (batch_size * self.nb_bins, self.nb_bins)
        )

        true_hic = batch.hic_matrix.detach().cpu()
        true_structure = batch.structure_matrix.detach().cpu()
        true_distance = batch.distance_matrix.detach().cpu()

        true_structure = torch.reshape(
            true_structure, (batch_size, self.nb_bins, self.embedding_size)
        )

        # Biological loss
        biological_loss = biological_loss_fct(
            pred_structure,
            true_structure,
            pred_distance,
            true_distance,
            self.nb_bins,
            batch_size,
        ).numpy()

        # Kabsch loss
        kabsch_loss = kabsch_loss_fct(
            pred_structure, true_structure, self.embedding_size, batch_size
        ).numpy()

        # Distance loss
        distance_loss = self.distance_loss_fct(pred_distance, true_distance).numpy()

        lddt_loss = loss_lddt(
            pred_structure, true_structure, logits, self.num_bins_logits
        )

        # To numpy
        true_hic = true_hic.numpy()

        pred_structure = pred_structure.numpy()
        true_structure = true_structure.numpy()

        pred_distance = pred_distance.numpy()
        true_distance = true_distance.numpy()

        # # Format results
        # true_hics = np.vstack(true_hics)
        #
        # pred_structures = np.vstack(pred_structures)
        # true_structures = np.vstack(true_structures)
        #
        # pred_distances = np.vstack(pred_distances)
        # true_distances = np.vstack(true_distances)
        #
        # # Compute mean losses
        # mean_biological_loss = np.mean(np.asarray(biological_loss).flatten())
        # mean_kabsch_loss = np.mean(np.asarray(kabsch_losses).flatten())
        # mean_distance_loss = np.mean(np.asarray(distance_losses).flatten())
        # mean_lddt_loss = np.mean(np.asarray(lddt_losses).flatten())

        self.log("biological_loss", biological_loss)
        self.log("kabsch_loss", kabsch_loss)
        self.log("distance_loss", distance_loss)
        self.log("lddt_losses", lddt_loss)


    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return self.optimizer(self.parameters(), **self.optimizer_kwargs)
