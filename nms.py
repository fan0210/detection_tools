#Created by fankai 2020/6/16 21:11, https://github.com/fan0210/detection_tools

import numpy as np
from polyiou import polyiou

def riounms(bboxes_per_img, classes, thresh=0.2):
    if len(bboxes_per_img) == 0:
        return []
    bboxes = []
    for bb in bboxes_per_img:
        x1,y1,x2,y2,x3,y3,x4,y4 = bb['bbox']
        score,cl,x1,y1,x2,y2,x3,y3,x4,y4 = float(bb['score']),float(classes.index(bb['class'])),float(x1),float(y1),float(x2),float(y2),float(x3),float(y3),float(x4),float(y4)
        bboxes.append([score,cl,x1,y1,x2,y2,x3,y3,x4,y4])
    bboxes = np.array(bboxes).astype('float32')
    bboxfiltered = []
    for c in range(15):
        dets = bboxes[bboxes[:,1]==c]
        if dets.shape[0] == 0:
            continue
        scores = dets[:, 0]
        polys = []
        areas = []
        for i in range(len(dets)):
            tm_polygon = polyiou.VectorDouble([float(dets[i][2]), float(dets[i][3]), float(dets[i][4]), float(dets[i][5]), float(dets[i][6]), float(dets[i][7]), float(dets[i][8]), float(dets[i][9])])
            polys.append(tm_polygon)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            ovr = []
            i = order[0]
            keep.append(i)
            for j in range(order.size - 1):
                iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
                ovr.append(iou)
            ovr = np.array(ovr)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        bboxfiltered.append(dets[keep])
    if len(bboxfiltered)==0:
        return []
    bboxfiltered = np.concatenate(bboxfiltered,axis=0)
    bboxes_out = []
    for box in bboxfiltered:
        bb={}
        bb['bbox']=[box[2],box[3],box[4],box[5],box[6],box[7],box[8],box[9]]
        bb['score']=box[0]
        bb['class']=classes[int(box[1])]
        bboxes_out.append(bb)
    return bboxes_out

