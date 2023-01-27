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
