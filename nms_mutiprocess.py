#Created by fankai 2020/6/16 20:28, https://github.com/fan0210/detection_tools

import os
import numpy as np
from polyiou import polyiou
from multiprocessing import Pool
from functools import partial

def riounms(pathsrc, classes, thresh):
    file = open(pathsrc,'r').readlines()
    os.remove(pathsrc)
    bboxes = []
    for bb in file:
        bb = bb.split()
        score,cl,x1,y1,x2,y2,x3,y3,x4,y4 = float(bb[0]),float(classes.index(str(bb[1]))),float(bb[2]),float(bb[3]),float(bb[4]),float(bb[5]),float(bb[6]),float(bb[7]),float(bb[8]),float(bb[9])
        bboxes.append([score,cl,x1,y1,x2,y2,x3,y3,x4,y4])
    bboxes = np.array(bboxes).astype('float32')
    bboxfiltered = []
    #bboxes = bboxes[bboxes[:,0]>0.05]
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
        return
    bboxfiltered = np.concatenate(bboxfiltered,axis=0)
    with open(pathsrc,'w') as fin:
        for box in bboxfiltered:
            score,cl,x1,y1,x2,y2,x3,y3,x4,y4 = box[0],classes[int(box[1])],box[2],box[3],box[4],box[5],box[6],box[7],box[8],box[9]
            fin.write(str(score)[0:5]+' '+cl+' '+str(int(x1))+' '+str(int(y1))+' '+str(int(x2))+' '+str(int(y2))+' '+str(int(x3))+' '+str(int(y3))+' '+str(int(x4))+' '+str(int(y4))+'\n')

def merge(dirsrc,dirdst,classes):
    files = os.listdir(dirsrc)
    fins = []
    for cl in classes:
        dstpath = 'Task1_'+cl+'.txt'
        fin = open(os.path.join(dirdst,dstpath),'a')
        fins.append(fin)
    for file in files:
        f = open(os.path.join(dirsrc,file),'r').readlines()
        oriname = file.split('.')[0]
        for bbox in f:
            bbox = bbox.split() 
            cl_id = classes.index(bbox[1])
            fins[cl_id].write(oriname+' '+bbox[0]+' '+bbox[2]+' '+bbox[3]+' '+bbox[4]+' '+bbox[5]+' '+bbox[6]+' '+bbox[7]+' '+bbox[8]+' '+bbox[9]+'\n')
    for fin in fins:
        fin.close()

def nms(files, classes, merge_for_dota=False, dota_merged_dir=None, iou_thresh=0.2, num_workers=8):
    filepathsrc = [os.path.join(files,f) for f in os.listdir(files)]
    worker = partial(riounms, classes=classes, thresh=iou_thresh)
    Pool(num_workers).map(worker, filepathsrc)
    if merge_for_dota:
        assert dota_merged_dir, "dota_merged_dir should not be None"
        if not os.path.exists(dota_merged_dir):
            os.mkdir(dota_merged_dir)
        merge(files,dota_merged_dir,classes)



