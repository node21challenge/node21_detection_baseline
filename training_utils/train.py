import training_utils.utils as utils
import numpy as np
import numpy as np
import torch
import torch.utils.data
import math
import torch.nn as nn

from torchvision.ops.boxes import _box_inter_union

"""
The following two functions are taken from: https://github.com/dbolya/yolact/blob/master/layers/box_utils.py
"""
def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def val_metrics(model, data_loader, device):
    """
    This is for calculating the mean iou of the current iteration
    :param model: Model
    :param data_loader: Validation data loader
    :param device:
    :return: the mean iou
    """
    for images, targets, img_name in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model.eval()

        list_targets_boxes = []
        list_targets_labels = []
        # Loop to take the bounding boxes and labels only
        for x in targets:
            list_targets_boxes.append(x['boxes'])
            list_targets_labels.append(x['labels'])
            if len(list_targets_labels) == 0:
                list_targets_labels.append(0)

        # Current model predictions
        pred_y = model(images)
        pred_y_boxes = []
        pred_y_labels = []
        for x in pred_y:
            pred_y_boxes.append(x['boxes'])
            pred_y_labels.append(x['labels'])

        loss_val = 0
        loss_val_total = 0
        count = 0
        # Calculate the iou for each prediction compared to each iou, then averages them
        for x in list_targets_boxes:
            for y in pred_y_boxes:
                loss_val += torch.sum(jaccard(x, y))
                if y.size(dim=0) > 0:
                    loss_val_total += loss_val.item() / y.size(dim=0)
        model.train()
    return loss_val_total


def train_one_epoch(model, optimizer, data_loader, val_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mean_iou', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    # criterion = torch.nn.functional.cross_entropy()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets, img_name in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Get validation metrics
        mean_iou = val_metrics(model, val_loader, device)
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(mean_iou=mean_iou)
