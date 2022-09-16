import torch
from .lddt_tools import lddt, get_confidence_metrics_post_soft
import torch.nn.functional as f
from scipy.special import softmax, log_softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration


class ModelWithTemperature(torch.nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, device, batch_size, nb_bins, embedding_size, num_bins_logits):
        self.device = device
        self.batch_size = batch_size
        self.nb_bins = nb_bins
        self.embedding_size = embedding_size
        self.num_bins_logits = num_bins_logits
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        pred_structure, pred_distance ,logits = self.model(input)
        return pred_structure, pred_distance, self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        #self.cuda()
        m = torch.nn.LogSoftmax(dim=1)
        nll_criterion = torch.nn.BCEWithLogitsLoss()
        #ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for data in valid_loader:
                data = data.to(self.device)
            
                pred_structure, pred_distance, logits = self.model(data.hic_matrix)
                pred_structure = pred_structure.detach().cpu()
            
                true_hic = data.hic_matrix.detach().cpu()
                true_structure = data.structure_matrix.detach().cpu()
                true_distance = data.distance_matrix.detach().cpu()
            
                true_structure = torch.reshape(true_structure, (self.batch_size, self.nb_bins, self.embedding_size))
                lddt_ca = lddt(
                        # Shape (batch_size, num_res, 3)
                        pred_structure,
                        # Shape (batch_size, num_res, 3)
                        true_structure,
                        # Shape (batch_size, num_res, 1)
                        cutoff=15.,
                        per_residue=True)
                num_bins = self.num_bins_logits
                bin_index = torch.floor(lddt_ca * num_bins).type(torch.torch.IntTensor)

                # protect against out of range for lddt_ca == 1
                bin_index = torch.minimum(bin_index, torch.tensor(num_bins, dtype=torch.int) - 1)
                label = f.one_hot(bin_index.to(torch.int64), num_classes=num_bins)
                label = torch.reshape(label, (self.batch_size*self.nb_bins, num_bins))
                
                logits = torch.reshape(logits, (self.batch_size*self.nb_bins, num_bins))
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list)#.type(torch.LongTensor)
            labels = torch.cat(labels_list).type(torch.float)
        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(m(logits.detach()), labels).item()
        print('Before temperature - NLL: %.3f' % (before_temperature_nll))

        optimizer = torch.optim.Adam([self.temperature], lr=0.01)
        
        def eval():
            optimizer.zero_grad()
            loss = 20*nll_criterion(m(self.temperature_scale(logits)), labels)
            print(loss)
            loss.backward()
            return loss
        for epoch in range(100):
            optimizer.step(eval)

        after_temperature_nll = nll_criterion(m(self.temperature_scale(logits)), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f' % (after_temperature_nll))

        return self



def isotonic_calibration(logits, labels, trussart_pred_logits, ):
    X = logits.detach().numpy()
    y = labels.numpy()
    pred_logits = trussart_pred_logits.detach().numpy()[0]
    trus = softmax(pred_logits, axis=-1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train = softmax(X_train, axis=-1)
    X_test = softmax(X_test, axis=-1)
    print(log_loss(Y_test, X_test))
    y_idunno = np.argmax(Y_train, axis=1)
    K = X_test.shape[1]
            
    for k in range(K):
        y_cal = np.array(y_idunno == k, dtype="int")
        if np.max(y_cal) == 0:
            continue
        isomodel = IsotonicRegression(y_min=0, y_max=1)
        isomodel.fit(X_train[:, k].reshape(-1, 1), y_cal) # Get only one column with probs for given class "k"

        X_train[:, k] = isomodel.predict(X_train[:, k])  # Predict new values based on the fittting
        X_test[:, k] = isomodel.predict(X_test[:, k])
        trus[:, k] = isomodel.predict(trus[:, k])

        idx_nan = np.where(np.isnan(X_train))
        X_train[idx_nan] = 0

        idx_nan = np.where(np.isnan(X_test))
        X_test[idx_nan] = 0

        idx_nan = np.where(np.isnan(trus))
        trus[idx_nan] = 0
    X_test = np.divide(X_test,np.sum(X_test, axis=1)[:, np.newaxis])

    print(log_loss(Y_test, X_test))

    confidence_metric_iso, pLDDT_iso = get_confidence_metrics_post_soft(trus)

    return confidence_metric_iso, pLDDT_iso

def beta_calibration(logits, labels, trussart_pred_logits):
    X = logits.detach().numpy()
    y = labels.numpy()
    pred_logits = trussart_pred_logits.detach().numpy()[0]
    trus = softmax(pred_logits, axis=-1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train = softmax(X_train, axis=-1)
    X_test = softmax(X_test, axis=-1)
    print(log_loss(Y_test, X_test))
    y_idunno = np.argmax(Y_train, axis=1)
    K = X_test.shape[1]
            
    # Go through all the classes
    for k in range(K):
        # Prep class labels (1 fixed true class, 0 other classes)
        y_cal = np.array(y_idunno == k, dtype="int")
        if np.max(y_cal) == 0:
            continue
        betamodel = BetaCalibration(parameters = "abm")
        betamodel.fit(X_train[:, k].reshape(-1, 1), y_cal) # Get only one column with probs for given class "k"

        X_train[:, k] = betamodel.predict(X_train[:, k])  # Predict new values based on the fittting
        X_test[:, k] = betamodel.predict(X_test[:, k])
        trus[:, k] = betamodel.predict(trus[:, k])

        idx_nan = np.where(np.isnan(X_train))
        X_train[idx_nan] = 0

        idx_nan = np.where(np.isnan(X_test))
        X_test[idx_nan] = 0

        idx_nan = np.where(np.isnan(trus))
        trus[idx_nan] = 0
    X_test = np.divide(X_test,np.sum(X_test, axis=1)[:, np.newaxis])

    print(log_loss(Y_test, X_test))

    confidence_metric_beta, plddt_beta = get_confidence_metrics_post_soft(trus)

    return confidence_metric_beta, plddt_beta