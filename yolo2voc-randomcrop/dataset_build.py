import os
import argparse
import glob
import shutil
import cv2
import numpy as np
from yolo2voc import *
from ima_preprocessing import * 

parser = argparse.ArgumentParser()

parser.add_argument('-s', action='store', dest='Image_Dir',
                    help='Directory of training images')

parser.add_argument('-l', action='store', dest='Label_Dir',
                    help='Directory of training labels')


parser.add_argument('-o', action='store', dest='Output_root',
                    help='root directory of output training data')

parser.add_argument('-val', action='store', dest='val_ratio',default=0.2,
                    help='train_validation split ratio')

parser.add_argument('-d_p', action='store', dest='dit_pro',default=0.1,
                    help='image files ratio for distorted_bounding_box_crop ')

parser.add_argument('-debug', action='store', dest='debug',default=False,
                    help='Debug flag, set True if you need to see the output image with bouding boxes')

results = parser.parse_args()

image_set =  glob.glob(results.Image_Dir+"/*.jpg")

imageNum =len(image_set)
print ("The total image number is %s")%(str(imageNum))

default_arr = np.random.choice(imageNum, imageNum, replace=False)
image_dict = {}
image_dict['train'] = default_arr[:int((1 - float(results.val_ratio))*imageNum)]
image_dict['val'] = default_arr[int((1 - float(results.val_ratio))*imageNum):]
print image_dict
print ("The training image number is %s")%(str(len(image_dict['train'])))
print ("The validation image number is %s")%(str(len(image_dict['val'])))

os.mkdir( results.Output_root)
os.mkdir( results.Output_root + "/train/")
os.mkdir( results.Output_root + "/val/")

D = {}
D["train"] = {}
D["val"] = {}
D["train"]["Image"] = results.Output_root + "/train/JPEGImages"
D["train"]["Label"] = results.Output_root + "/train/Annotations"
D["val"]["Image"] = results.Output_root + "/val/JPEGImages"
D["val"]["Label"] = results.Output_root + "/val/Annotations"
os.mkdir(D["train"]["Image"])
os.mkdir(D["train"]["Label"])
os.mkdir(D["val"]["Image"])
os.mkdir(D["val"]["Label"])

for p in ["train","val"]:
    file_id = 0
    for image_path_id in (image_dict[p]):
        try:
            image_path = image_set[image_path_id].strip()
            #print image_path
            image_name = image_path[image_path.rfind("/")+1:-4]
            label_path = results.Label_Dir +"/%s.txt"%image_name
            #print label_path
            out_label_path = D[p]["Label"] + "/%06d.xml"%int(file_id)
            if os.path.isfile(label_path):
                image = cv2.imread(image_path)
                im_h,im_w,im_d = image.shape
                with open(label_path) as f:
                    object_infos = f.readlines()
        
                labels = []
                boxes = []
                for _object in object_infos:
                    _object = _object.strip().split()
                    xc = float(_object[1])
                    yc = float(_object[2])
                    w = float(_object[3])
                    h = float(_object[4])
                    x1 = int((xc - 0.5*w)*float(im_w))
                    x2 = int((xc + 0.5*w )*float(im_w))
                    y1 = int((yc - 0.5*h)*float(im_h))
                    y2 = int((yc + 0.5*h)*float(im_h))
                    if (x1<x2) and (y1<y2):
                        rect = {}
                        rect["x1"] = x1
                        rect["x2"] = x2
                        rect["y1"] = y1
                        rect["y2"] = y2
                        labels.append(_object[0])
                        boxes.append(rect)
                    else:
                        pass

                yolo2voc(file_id,im_w,im_h,im_d,labels,boxes,out_label_path)
                if (results.debug):
                    for rect in boxes:
                        cv2.rectangle(image, (rect["x1"], rect["y1"]), (rect["x2"], rect["y2"]), (255,0,0), 2)
                    cv2.imshow("test",image)
                    cv2.waitKey(0)
                new_image_path = D[p]["Image"] + "/%06d.jpg"%int(file_id)
                shutil.copyfile(image_path,new_image_path)
                file_id += 1
                print file_id

                distort_decision = np.random.choice(["Y","N"], 1, p=[float(results.dit_pro),1-float(results.dit_pro)])

                if p == "train" and distort_decision == "Y" and len(labels)>0:
                    dist_img,dist_labels,dist_boxes = random_crop(image,labels,boxes)
                    out_label_path = D[p]["Label"] + "/%06d.xml"%int(file_id) 
                    im_h,im_w,im_d = dist_img.shape
                    yolo2voc(file_id,im_w,im_h,im_d,dist_labels,dist_boxes,out_label_path)
                    
                    new_image_path = D[p]["Image"] + "/%06d.jpg"%int(file_id)
                    if (results.debug):
                        for rect in dist_boxes:
                            cv2.rectangle(dist_img, (rect["x1"], rect["y1"]), (rect["x2"], rect["y2"]), (255,0,0), 2)
                        cv2.imshow("test",dist_img)
                        cv2.waitKey(0)
                
                    cv2.imwrite(new_image_path,dist_img)
                    file_id += 1
                    print file_id
                    print distort_decision

            else:
                pass
        except:
            pass
