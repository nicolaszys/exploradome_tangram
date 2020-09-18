import cv2
import random
import numpy as np
import tensorflow as tf
import operator
import sys





def output_box(all_boxes, all_preds, boxes, bboxes, class_names):
    
     #  matching bounding boxes non formated from combined_non_max_suppression, with their respective classe predictions (pred_conf tensor) using index

    
    idx=[np.where(all_boxes==boxes.numpy()[0][i])[0][0] for i in range(len(boxes.numpy()[0]))]
    proba_bboxes=[all_preds[i] for i in idx]


    # initialization 
    boxes_and_preds=[] # list of bboxe coordinates and classes predictions  for each bboxes
    proba_sum=0

    # loop over bboxes (formatted)
    for i in range(len(bboxes)):
        coord=bboxes[i]
        # bbox coordinates : (x1,y1) -->  upper left corner of the bbox ; (x2,y2) -->  bottom right corner of the bbox
        coord_i={'x1': coord[0],'y1':  coord[1] , 'x2': coord[2], 'y2':  coord[3]}
        #c1, c2, c3, c4= (coord[0], coord[1]), (coord[2], coord[3]), (coord[0], coord[3]), (coord[2], coord[1])

        # Making sum of probas equals to 100%
        sum_proba_i=sum(proba_bboxes[i])
        proba_i=  {class_names[j]:proba_bboxes[i][j]*100/sum_proba_i for j in range(len(class_names))}

        # dictionnary filling 
        dic = {'object_position': coord_i, 'object_prediction':proba_i}
        boxes_and_preds.append(dic)

    return boxes_and_preds

        

def sort_by(x, max_classes):
    for i in range(len(x)):
        x[i]['object_prediction']= dict(sorted(x[i]['object_prediction'].items(), key=operator.itemgetter(1),reverse=True)[:max_classes])
    return x


def pointInRect(point,rect):
    x1, y1, x2, y2 = rect
    x, y = point
    if (x1 < x and x < x2) :
        if (y1 < y and y < y2):
            return True
    return False

def format_game_areas(game_areas, image_height, image_width):
    formatted_game_areas=[]
    for coord in game_areas:
        x1 = int(coord[0] * image_width)
        y1 = int(coord[1] * image_height)
        x2 = int(coord[2] * image_width)
        y2 = int(coord[3] * image_height)
        formatted_game_areas.append([x1, y1, x2, y2])
    return formatted_game_areas


def filter_game_areas(boxes_and_preds,game_areas,k):
    output=[]
    area_dic={'gauche':None,'droite':None,'centre':None}
    for i in range(len(boxes_and_preds)):
    
        # object bounding box coordinates
        coords=boxes_and_preds[i]['object_position']
        
        #corners c1 --> upper left, c2 --> lower right, c3 --> lower left, c4 --> upper right
        [c1,c2,c3,c4]=[(coords['x1'],coords['y1']),(coords['x2'],coords['y2']), (coords['x1'],coords['y2']),(coords['x2'],coords['y1'])]
        
        # loop over the different game areas (ie : left, right, center)
        for area,area_name in zip(game_areas,area_dic.keys()):
            
            # check  if the object bounding box is in the game area 
            inrect_results=[pointInRect(corner,area) for corner in [c1,c2,c3,c4]] # return a list of True or False depending if the corner belongs to the area
            
            # define a hard margin (k=4) or a soft margin (k <4) condition, in order to decide if we keep object that oversteep the game area 
            if sum(inrect_results) >=  k:
                area_dic[area_name]={'object_position':boxes_and_preds[i]['object_position'], 'object_prediction':boxes_and_preds[i]['object_prediction']}
    output.append(area_dic)
    return output
    


    
