
__credits__ = ['Pavel Yakubovskiy, https://github.com/qubvel/segmentation_models.pytorch']

import torch
import utils.volutils as volutils

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def weighted_dice_score(pr, gt, beta=1, eps=1e-7, boundary_weight=10, boundary_thickness=1,
                        device='cpu', threshold=None, ignore_channels=None):
    """Calculate weighted dice-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
        weight: weight factor for border pixels
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = gt * pr
    fp = pr - tp
    fn = gt - tp

    # calculate loss independently for both types of pixels
    borders = volutils.easy_borders(gt, thickness=boundary_thickness)
    w = torch.Tensor(borders*(boundary_weight-1) + 1)
    w = w.to(device)

    nominator = ((1 + beta ** 2) * torch.sum(tp*w) + eps)
    denominator = ((1 + beta ** 2) * torch.sum(tp*w) + beta ** 2 * torch.sum(fn*w) + torch.sum(fp*w) + eps)
    score = nominator / denominator

    return score


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]
    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score


def min_euclidean(y_pr, y_gt):
    """Sum over euclidean distances between each grid point and closes predicted points
    Args:
     y_pr (torch.Tensor): predicted tensor
     y_gt (torch.Tensor):  ground truth tensor
    Returns:
     float: euclidean loss
    """
    # reshape to get spatial dimensions
    y_gt = y_gt.reshape((y_gt.shape[0], -1, 2))
    y_pr = y_pr.reshape((y_pr.shape[0], -1, 2))

    # get all distances from all ground truth to all predicted points
    dist = torch.norm(torch.unsqueeze(y_gt, 2) - torch.unsqueeze(y_pr, 1), p=2, dim=3)

    # get minimal distance for each ground truth point (reduce over the dimension for the predicted points)
    dist, _ = dist.min(dim=2)

    loss = dist.mean()
    return dist.mean()


def wasserstein_distance(p, p_hat):
    """Computes Wasserstein distance between to sets of points

    Args:
        p, p_hat: (batch_size x N_point x 2) arrays of points

    Returns:
        distance: Wasserstein distance
    """
    assert p.shape == p_hat.shape
    batch_size, n_points, dim = p.shape

    diff12 = torch.norm(p.reshape((batch_size, n_points, -1, dim))
                        - p_hat.reshape((batch_size, -1, n_points, dim)), p=2, dim=-1)
    diff21 = torch.norm(p_hat.reshape((batch_size, n_points, -1, dim))
                        - p.reshape((batch_size, -1, n_points, dim)), p=2, dim=-1)
    diff12, _ = diff12.min(dim=-1)
    diff21, _ = diff21.min(dim=-1)
    distance = diff12.mean() + diff21.mean()
    return distance


def total_error(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    n_pixels = pr.numel()
    score = (fn + fp) / n_pixels

    return score


def false_positive_rate(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    #fn = torch.sum(gt) - tp
    n = torch.sum(1-gt)
    tn = n - fp

    score = fp / (fp + tn)

    return score


def false_negative_rate(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = fn / (fn + tp)

    return score