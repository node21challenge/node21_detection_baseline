 
def get_NonMaxSup_boxes(pred_dict):
        scores = pred_dict['scores']
        boxes = pred_dict['boxes']
        lambda_nms = 0.3

        out_scores = []
        out_boxes = []
        for ix, (score, box) in enumerate(zip(scores,boxes)):
            discard = False
            for other_box in out_boxes:
                if intersection_over_union(box, other_box) > lambda_nms:
                    discard = True
                    break
            if not discard:
                out_scores.append(score)
                out_boxes.append(box)
        return {'scores':out_scores, 'boxes':out_boxes}
    
# Source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou