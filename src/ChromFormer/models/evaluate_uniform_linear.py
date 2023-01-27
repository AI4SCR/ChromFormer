def evaluate_uniform_linear(
    loader,
    model,
    device,
    batch_size,
    nb_bins,
    embedding_size,
    distance_loss_fct
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
