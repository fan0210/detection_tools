# Created by fankai 2020/6/8 23:15, https://github.com/fan0210/detection_tools/

import numpy as np
import cv2
import math

def poly_to_rect(polygons):
    """
    polygons:
        type=np.array.float32
        shape=(n_poly,8)
    return:
        type=np.array.float32
        shape=(n_poly,8)
    """
    rects_min_area = []
    for poly in polygons:
        x1,y1,x2,y2,x3,y3,x4,y4 = poly
        (r_x1,r_y1),(r_x2,r_y2),(r_x3,r_y3),(r_x4,r_y4) = \
            cv2.boxPoints(cv2.minAreaRect(np.array([[int(x1),int(y1)],[int(x2),int(y2)],[int(x3),int(y3)],[int(x4),int(y4)]])))
        rects_min_area.append([r_x1,r_y1,r_x2,r_y2,r_x3,r_y3,r_x4,r_y4])
    rects_min_area = np.array(rects_min_area)
    w_rects = np.sqrt((rects_min_area[:,0]-rects_min_area[:,2])**2+(rects_min_area[:,1]-rects_min_area[:,3])**2)
    h_rects = np.sqrt((rects_min_area[:,2]-rects_min_area[:,4])**2+(rects_min_area[:,3]-rects_min_area[:,5])**2)
    mask_x1_eq_x2 = rects_min_area[:,0]==rects_min_area[:,2]
    angle_w = np.where(mask_x1_eq_x2,0.5,np.arctan((rects_min_area[:,1]-rects_min_area[:,3])/
                                                  (np.where(mask_x1_eq_x2,0.0001,rects_min_area[:,2]-rects_min_area[:,0])))/math.pi)
    mask_x2_eq_x3 = rects_min_area[:,2]==rects_min_area[:,4]
    angle_h = np.where(mask_x2_eq_x3,0.5,np.arctan((rects_min_area[:,3]-rects_min_area[:,5])/
                                                  (np.where(mask_x2_eq_x3,0.0001,rects_min_area[:,4]-rects_min_area[:,2])))/math.pi)
    angle = np.where(w_rects>h_rects,angle_w,angle_h)
    angle = np.where(angle<0,angle+1,angle)
    mx_1,mx_2,mx_3,mx_4,my_1,my_2,my_3,my_4 = \
        (polygons[:,0]+polygons[:,2])/2,(polygons[:,2]+polygons[:,4])/2,(polygons[:,4]+polygons[:,6])/2,(polygons[:,6]+polygons[:,0])/2,\
        (polygons[:,1]+polygons[:,3])/2,(polygons[:,3]+polygons[:,5])/2,(polygons[:,5]+polygons[:,7])/2,(polygons[:,7]+polygons[:,1])/2
    wh = np.concatenate([np.sqrt((mx_1-mx_3)**2+(my_1-my_3)**2)[:,None],np.sqrt((mx_2-mx_4)**2+(my_2-my_4)**2)[:,None]],axis=-1)
    mask_mx1_eq_mx3 = mx_1==mx_3
    angle_mw = np.where(mask_mx1_eq_mx3,0.5,np.arctan((my_1-my_3)/
                                                  (np.where(mask_mx1_eq_mx3,0.0001,mx_3-mx_1)))/math.pi)
    mask_mx2_eq_mx4 = mx_2==mx_4
    angle_mh = np.where(mask_mx2_eq_mx4,0.5,np.arctan((my_2-my_4)/
                                                  (np.where(mask_mx2_eq_mx4,0.0001,mx_4-mx_2)))/math.pi)
    angle_m = np.where(wh[:,0]>wh[:,1],angle_mw,angle_mh)
    angle_m = np.where(angle_m<0,angle_m+1,angle_m)
    angle_err = np.sin(np.abs(angle-angle_m)*math.pi)
    angle_out = np.where(angle_err>math.sin(math.pi/4),angle-0.5,angle)
    angle_out = np.where(angle_out<0,angle_out+1,angle_out)
    angle_out = angle_out*math.pi
    cx_out = polygons[:,0::2].sum(axis=-1)/4
    cy_out = polygons[:,1::2].sum(axis=-1)/4
    w_out = np.max(wh,axis=-1)
    h_out = np.min(wh,axis=-1)
    half_dl = np.sqrt(w_out*w_out+h_out*h_out)/2
    a1 = math.pi-(math.pi-angle_out+np.arccos(w_out/2/half_dl))
    a2 = 2*np.arctan(h_out/w_out)+a1
    x1 = half_dl*np.cos(a2)
    y1 = -half_dl*np.sin(a2)
    x2 = half_dl*np.cos(a1)
    y2 = -half_dl*np.sin(a1)
    x3,y3,x4,y4 = -x1,-y1,-x2,-y2

    return np.stack([x1+cx_out,y1+cy_out,x2+cx_out,y2+cy_out,x3+cx_out,y3+cy_out,x4+cx_out,y4+cy_out]).transpose()

if __name__ == '__main__':
    import os
    import shapely.geometry as shgeo

    imgpath = r'F:\polytorect\P0049.png'
    labelpath = r'F:\polytorect\P0049.txt'

    img = cv2.imread(imgpath)

    files = open(labelpath,'r').readlines()
    bboxes_poly = np.array([bbox.split()[0:8] for bbox in files if len(bbox.split())==10],dtype='float32')

    bboxes_min_area = []
    for poly in bboxes_poly:
        x1,y1,x2,y2,x3,y3,x4,y4 = poly
        (r_x1,r_y1),(r_x2,r_y2),(r_x3,r_y3),(r_x4,r_y4) = \
            cv2.boxPoints(cv2.minAreaRect(np.array([[int(x1),int(y1)],[int(x2),int(y2)],[int(x3),int(y3)],[int(x4),int(y4)]])))
        bboxes_min_area.append([r_x1,r_y1,r_x2,r_y2,r_x3,r_y3,r_x4,r_y4])
    bboxes_min_area = np.array(bboxes_min_area)

    bboxes_rect = poly_to_rect(bboxes_poly)
    colors = [(0,255,0),(0,0,255),(255,0,0)]
    for i in range(len(bboxes_rect)):
        bboxes = [bboxes_rect[i],bboxes_poly[i],bboxes_min_area[i]]
        bboxes_shego = []
        for j,bb in enumerate(bboxes):
            x1,y1,x2,y2,x3,y3,x4,y4 = bb.astype('int32')
            cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),colors[j],1,16)
            cv2.line(img,(int(x2),int(y2)),(int(x3),int(y3)),colors[j],1,16)
            cv2.line(img,(int(x3),int(y3)),(int(x4),int(y4)),colors[j],1,16)
            cv2.line(img,(int(x4),int(y4)),(int(x1),int(y1)),colors[j],1,16)
            bboxes_shego.append(shgeo.Polygon([(x1, y1),(x2, y2),(x3, y3),(x4, y4)]))

        rect_refined,polygon,rect_min_area=bboxes_shego
        area_rect_refined = rect_refined.area
        area_polygon = polygon.area
        area_rect_min_area = rect_min_area.area

        intersec_rectrefined_ploy = rect_refined.intersection(polygon)
        intersec_rectminarea_poly = rect_min_area.intersection(polygon)
        iou1 = intersec_rectrefined_ploy.area/(area_rect_refined+area_polygon-intersec_rectrefined_ploy.area)
        iou2 = intersec_rectminarea_poly.area/(area_rect_min_area+area_polygon-intersec_rectminarea_poly.area)
        print(iou1,iou2)

    cv2.imwrite('im.png',img)


