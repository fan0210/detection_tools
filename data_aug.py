"""
ң��ͼ��Ŀ�����������䣨������ǿ����������OBB��HBB
"""

import cv2
import numpy as np
import random
import copy

network_input_size = 800 #���������С

def HBB(img, bboxes):
    """
    HBBĿ�����������䡣

    ���룺
        img:��ͨ��unsigned char(uint8)����ͼ������
        bboxes:��ͼ���Ӧ��HBB*n, ÿ��HBB��ʽΪ[xmin,ymin,xmax,ymax], bboxes��shapeΪ(n,4)������Ϊnp.array.float32
    �����
        img_crop:��ǿ���ͼ��(float���ͣ���Χ0~1, ��ͨ��)
        bboxes_crop:��Ӧ��bboxes gt������������ʽ��ͬ
    """

    #1.��ת�任����ת�Ƕ�0,90,180,270
    random_angle = -random.randint(0,3)*90
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, random_angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    rotated_img = cv2.warpAffine(img, M, (nW, nH))
    r_h, r_w = rotated_img.shape[:2]
    bboxes_rotated = []
    for i in range(2):
        bboxes_rotated.append(np.matmul(M,np.concatenate([bboxes[:,i*2:(i+1)*2].T,np.ones([1,bboxes.shape[0]])],axis=0)).T)
    bboxes_rotated = np.concatenate(bboxes_rotated,axis=1)
    bboxes_rotated_copy = copy.deepcopy(bboxes_rotated)
    bboxes_rotated[:,0] = np.min(np.concatenate([bboxes_rotated_copy[:,0][:,np.newaxis],bboxes_rotated_copy[:,2][:,np.newaxis]],axis=-1),axis=-1)
    bboxes_rotated[:,1] = np.min(np.concatenate([bboxes_rotated_copy[:,1][:,np.newaxis],bboxes_rotated_copy[:,3][:,np.newaxis]],axis=-1),axis=-1)
    bboxes_rotated[:,2] = np.max(np.concatenate([bboxes_rotated_copy[:,0][:,np.newaxis],bboxes_rotated_copy[:,2][:,np.newaxis]],axis=-1),axis=-1)
    bboxes_rotated[:,3] = np.max(np.concatenate([bboxes_rotated_copy[:,1][:,np.newaxis],bboxes_rotated_copy[:,3][:,np.newaxis]],axis=-1),axis=-1)

    #2.ͼ�����(random crop)
    crop_xmin = int(random.uniform(0,0.2*np.min(bboxes_rotated[:,0])))
    crop_ymin = int(random.uniform(0,0.2*np.min(bboxes_rotated[:,1])))
    crop_xmax = int(random.uniform(np.max(bboxes_rotated[:,2])+(r_w-np.max(bboxes_rotated[:,2]))*0.8,r_w))
    crop_ymax = int(random.uniform(np.max(bboxes_rotated[:,3])+(r_h-np.max(bboxes_rotated[:,3]))*0.8,r_h))

    img_crop = rotated_img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

    bboxes_crop=copy.deepcopy(bboxes_rotated)
    bboxes_crop[:,[0,2]] = (bboxes_crop[:,[0,2]]-crop_xmin)*network_input_size/img_crop.shape[1]
    bboxes_crop[:,[1,3]] = (bboxes_crop[:,[1,3]]-crop_ymin)*network_input_size/img_crop.shape[0]
    img_crop = cv2.resize(img_crop,(network_input_size, network_input_size)).astype('float32')/255.

    #3.����������ҷ�ת
    if random.random()<0.5:
        img_crop = img_crop[::-1]
        bboxes_crop[:,[3,1]] = network_input_size-bboxes_crop[:,[1,3]]
    if random.random()<0.5:
        img_crop = img_crop[:,::-1]
        bboxes_crop[:,[2,0]] = network_input_size-bboxes_crop[:,[0,2]]

    #4.ͼ����������
    rs = np.array([[random.random(),random.random()] for channel in range(3)])
    for c in range(3):
        if rs[c,0]>0.5:
            img_crop[:,:,c]=img_crop[:,:,c]*(1+(1-rs[c,0])/10.)+(0.5-rs[c,1])/20.0
        else:
            img_crop[:,:,c]=img_crop[:,:,c]*(1-rs[c,0]/10.)+(0.5-rs[c,1])/20.0
    img_crop = np.clip(img_crop,0.0,1.0)

    #5.����ҶȻ�(0.3�ĸ���)
    if random.random()<0.3:
        select_channel = img_crop[:,:,random.randint(0,2)]
        for i in range(3):
            img_crop[:,:,i] = select_channel

    return img_crop, bboxes_crop

def OBB(img, bboxes):
    """
    OBBĿ�����������䡣

    ���룺
        img:��ͨ��unsigned char(uint8)����ͼ�����ݣ�width��height�����
        bboxes:��ͼ���Ӧ��OBB*n, ÿ��OBB��ʽΪ[x1,y1,x2,y2,x3,y3,x4,y4], bboxes��shapeΪ(n,8)������Ϊnp.array.float32
    �����
        img_crop:��ǿ���ͼ��(float���ͣ���Χ0~1, ��ͨ��)
        bboxes_crop:��Ӧ��bboxes gt��������Ϊ��ת���Σ���ʽ�Ժ������ʽ��ͬ
    """

    #1.��ת�任����ת�Ƕ�Ϊ0~360�����ʵ�鷢�������ת��Ȼ�����ڱߵ��Ǳȹ̶��ĸ��Ƕ���ת���Ȼ���0.5�������ҵ�������
    random_angle = -random.random()*360
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, random_angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    rotated_img = cv2.warpAffine(img, M, (nW, nH))
    size = rotated_img.shape[0]

    bboxes_rotated = []
    for i in range(4):
        bboxes_rotated.append(np.matmul(M,np.concatenate([bboxes[:,i*2:(i+1)*2].T,np.ones([1,bboxes.shape[0]])],axis=0)).T)
    bboxes_rotated = np.concatenate(bboxes_rotated,axis=1)

    #2.ͼ�����(random crop)
    min_x = np.min([np.min(bboxes_rotated[:,0]),np.min(bboxes_rotated[:,2]),np.min(bboxes_rotated[:,4]),np.min(bboxes_rotated[:,6])])
    min_y = np.min([np.min(bboxes_rotated[:,1]),np.min(bboxes_rotated[:,3]),np.min(bboxes_rotated[:,5]),np.min(bboxes_rotated[:,7])])
    max_x = np.max([np.max(bboxes_rotated[:,0]),np.max(bboxes_rotated[:,2]),np.max(bboxes_rotated[:,4]),np.max(bboxes_rotated[:,6])])
    max_y = np.max([np.max(bboxes_rotated[:,1]),np.max(bboxes_rotated[:,3]),np.max(bboxes_rotated[:,5]),np.max(bboxes_rotated[:,7])])
    if max_y-min_y>max_x-min_x:
        crop_ymin = int(random.uniform(0, 0.5*min_y))
        crop_ymax = int(random.uniform(max_y+0.5*(size-max_y), size))
        crop_size = crop_ymax - crop_ymin
        crop_xmin = int(random.uniform(max(0,max_x-crop_size),min(min_x,size-crop_size)))
        crop_xmax = crop_xmin+crop_size
    else:
        crop_xmin = int(random.uniform(0, 0.5*min_x))
        crop_xmax = int(random.uniform(max_x+0.5*(size-max_x), size))
        crop_size = crop_xmax - crop_xmin
        crop_ymin = int(random.uniform(max(0,max_y-crop_size),min(min_y,size-crop_size)))
        crop_ymax = crop_ymin+crop_size
    img_crop = rotated_img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
    bboxes_crop = copy.deepcopy(bboxes_rotated)
    bboxes_crop[:,[0,2,4,6]] = bboxes_crop[:,[0,2,4,6]]-crop_xmin
    bboxes_crop[:,[1,3,5,7]] = bboxes_crop[:,[1,3,5,7]]-crop_ymin    

    bboxes_crop = bboxes_crop*network_input_size/crop_size
    bboxes_mean_area = []
    for bbox in bboxes_crop:
        x1,y1,x2,y2,x3,y3,x4,y4 = bbox
        rect = cv2.minAreaRect(np.array([[int(x1),int(y1)],[int(x2),int(y2)],[int(x3),int(y3)],[int(x4),int(y4)]]))
        rect = cv2.boxPoints(rect)
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = rect
        bboxes_mean_area.append([x1,y1,x2,y2,x3,y3,x4,y4])
        
    img_crop = cv2.resize(img_crop,(network_input_size, network_input_size)).astype('float32')/255.

    #3.����������ҷ�ת
    if random.random()<0.5:
        img_crop = img_crop[::-1]
        bboxes_mean_area[:,[1,3,5,7]] = network_input_size-bboxes_mean_area[:,[1,3,5,7]]
    if random.random()<0.5:
        img_crop = img_crop[:,::-1]
        bboxes_mean_area[:,[0,2,4,6]] = network_input_size-bboxes_mean_area[:,[0,2,4,6]]

    #4.ͼ����������
    rs = np.array([[random.random(),random.random()] for channel in range(3)])
    for c in range(3):
        if rs[c,0]>0.5:
            img_crop[:,:,c]=img_crop[:,:,c]*(1+(1-rs[c,0])/10.)+(0.5-rs[c,1])/20.0
        else:
            img_crop[:,:,c]=img_crop[:,:,c]*(1-rs[c,0]/10.)+(0.5-rs[c,1])/20.0
    img_crop = np.clip(img_crop,0.0,1.0)

    #5.����ҶȻ�(0.3�ĸ���)
    if random.random()<0.3:
        select_channel = img_crop[:,:,random.randint(0,2)]
        for i in range(3):
            img_crop[:,:,i] = select_channel

    return img_crop, np.array(bboxes_mean_area)




