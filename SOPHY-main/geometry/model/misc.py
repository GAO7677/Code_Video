import torch
import torch.nn.functional as F
from typing import List, Literal, Optional
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, confusion_matrix


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist -= 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def prepare_mat_inputs(mat_Es, mat_nus, mat_sigmas, mat_phis, mat_rhos, mat_mmid, args):
    mat_features = list()
    if args.requires_e:
        mat_features.append(mat_Es.clamp(min=0.))
    if args.requires_nu:
        mat_features.append(mat_nus)
    if args.requires_sigma:
        mat_features.append(mat_sigmas.clamp(min=0.))
    if args.requires_phi:
        mat_features.append(mat_phis)
    if args.requires_rho:
        mat_features.append(mat_rhos.clamp(min=0.))
    if args.num_mmid > 0:
        mat_features.append(mat_mmid)
    mat_features = torch.stack(mat_features, dim=-1)

    return mat_features


def consistency_loss(embeddings, labels, normalization=True, margin=None):
    """
    Calculate the consistency loss for embeddings with positive labels.

    Args:
        embeddings: pytorch Variable, with shape [n, d]
        labels: pytorch LongTensor, with shape [n]
        normalization: bool, whether to normalize the embeddings
        margin: float, distance below which the loss is 0
    Returns:
        consistency_loss: pytorch Variable, with shape []
    """

    if normalization:
        embeddings = normalize(embeddings, axis=-1)
    dist_mat = euclidean_dist(embeddings, embeddings)
    is_pos = labels.expand_as(dist_mat).eq(labels.expand_as(dist_mat).t())
    indices_not_equal = ~torch.eye(
        dist_mat.shape[0], dtype=bool, device=embeddings.device)
    is_pos = torch.logical_and(is_pos, indices_not_equal)

    # only consider the distances between positive pairs
    dist_pos = dist_mat * is_pos.float()
    if margin is not None:
        dist_pos = torch.clamp(dist_pos - margin, min=0.0)
    return dist_pos.sum() / is_pos.float().sum()


def material_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    occupancy: torch.Tensor,
    use_empty: bool,
    occ_version: Literal['cross_entropy', 'binary', 'gumbel'] = 'cross_entropy',
    targets_all_possible: Optional[torch.Tensor] = None,
    tau_gumbel: float = 1.0,
    eps: float = 1e-8,
):
    """ 
    Calculate the material loss for a batch of inputs and targets.

    Args:
        inputs: torch.Tensor, with shape [N, P, C]
        targets: torch.Tensor, with shape [N, P], categorical material labels
        occupancy: torch.Tensor, with shape [N, P].
        use_empty: bool, whether to use empty class.
        occ_version: str, version of material loss. Default is 'cross_entropy'.
        targets_all_possible: torch.Tensor, with shape [N, P, C], targets for all possible materials.
        tau_gumbel: float, temperature for gumbel softmax.
        eps: float, small value to avoid numerical instability.

    Returns:
        loss: material loss for occupied and empty areas.
    """
    N, P, _ = inputs.shape
    inputs = inputs.reshape(N*P, -1)
    targets = targets.reshape(N*P)
    occupancy = occupancy.reshape(N*P)

    # 1. material loss for occupied areas
    occ_inputs = inputs[occupancy==1]
    occ_targets = targets[occupancy==1]
    # if use_empty:
    #     occ_inputs = occ_inputs[:, 1:]
    #     occ_targets = occ_targets - 1
    if occ_version == 'gumbel':
        targets_gumbel = targets_all_possible.reshape(N*P, -1)
        targets_gumbel = targets_gumbel[occupancy==1]
        # if use_empty:
        #     targets_gumbel = targets_gumbel[:, 1:]
        z_hat = F.gumbel_softmax(occ_inputs, tau=tau_gumbel, hard=False)
        log_z_hat = torch.log(z_hat + eps)
        masked_log_z_hat = targets_gumbel * log_z_hat
        occ_loss = - torch.sum(masked_log_z_hat, dim=-1) / torch.sum(targets_gumbel.float(), dim=-1)
        occ_loss = torch.mean(occ_loss)
    elif occ_version == 'binary':
        targets_binary = targets_all_possible.reshape(N*P, -1)
        targets_binary = targets_binary[occupancy==1]
        occ_loss = F.binary_cross_entropy_with_logits(occ_inputs, targets_binary.float())
    else:
        occ_loss = F.cross_entropy(occ_inputs, occ_targets)

    # 2. material loss for empty areas
    if use_empty:
        emp_inputs = inputs[occupancy==0]
        emp_targets = targets[occupancy==0]
        emp_loss = F.cross_entropy(emp_inputs, emp_targets)
    else:
        emp_loss = 0.
    
    return occ_loss, emp_loss


def material_accuracy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    occupancy: torch.Tensor,
    reduction: Literal['mean', 'none'] = 'mean'
):
    """
    Calculate the material accuracy for a batch of inputs and targets.

    Args:
        inputs: torch.Tensor, **one hot** tensor with shape [N, P, C].
        targets: torch.Tensor, with shape [N, P, C].
        occupancy: torch.Tensor, with shape [N, P].

    Returns:
        accuracy: material accuracy
    """
    mat = inputs.bool() & targets.bool()
    # (N, P)
    mat = mat.sum(dim=-1)

    # only compute accuracy for occupied voxels
    batch_acc = (mat * occupancy).sum(dim=1) / occupancy.sum(dim=1)
    if reduction == 'mean':
        return batch_acc.mean()
    else:
        return batch_acc    # (N,)


def find_best_threshold(y_trues, y_preds):
    # print("Finding best threshold...")
    best_thre = 0.5
    best_metrics = None
    candidate_thres = list(np.unique(np.sort(y_preds)))
    for thre in candidate_thres:
        metrics = material_statistics(y_trues, y_preds, threshold=thre)
        if best_metrics is None:
            best_metrics = metrics
            best_thre = thre
        elif metrics.get("ACER") < best_metrics.get("ACER"):
            best_metrics = metrics
            best_thre = thre
    # print(f"Best threshold is {best_thre}")
    return best_thre, best_metrics


def material_statistics(y_trues, y_preds, threshold=0.5):
    assert len(y_trues) == len(y_preds), \
        f"Length of y_trues ({len(y_trues)}) and y_preds ({len(y_trues)}) should be equal."
    metrics = dict()

    fpr, tpr, thresholds = roc_curve(y_trues, y_preds, pos_label=1)
    metrics.update({"eer": brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)})
    metrics.update({"thre": float(interp1d(fpr, thresholds)(metrics.get("eer")))})

    if threshold == 'best':
        _, best_metrics = find_best_threshold(y_trues, y_preds)
        return best_metrics

    elif threshold == 'auto':
        threshold = metrics.get("thre")

    else:
        metrics.update({"thre": threshold})

    prediction = (np.array(y_preds) > threshold).astype(int)

    res = confusion_matrix(y_trues, prediction)
    TP, FN = res[0, :]
    FP, TN = res[1, :]
    metrics.update({"accuracy": (TP + TN) / len(y_trues)})
    metrics.update({"precision": float(TP / (TP + FP))})
    metrics.update({"recall": float(TP / (TP + FN))})

    assert TP + FP + TN + FN == len(y_trues), \
        "Sum of confusion matrix should be equal to the number of samples."

    metrics.update({"tp": TP})
    metrics.update({"fp": FP})
    metrics.update({"tn": TN})
    metrics.update({"fn": FN})
    
    return metrics
