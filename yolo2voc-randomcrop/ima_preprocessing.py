import sys
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import dataset_factory
import numpy as np
import tf_extended as tfe

# Some training pre-processing parameters.
BBOX_CROP_OVERLAP = 0.5         # Minimum overlap to keep a bbox after cropping.
MIN_OBJECT_COVERED = 0.25
CROP_RATIO_RANGE = (0.6, 1.67)  # Distortion ratio during cropping.
EVAL_SIZE = (300, 300)

slim = tf.contrib.slim

def box_thred(x):
    if x < 0:
        return 0
    elif  x > 1:
        return 1
    else:
        return x

def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                clip_bboxes=True,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes = tfe.bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = tfe.bboxes_filter_overlap(labels, bboxes,
                                                   threshold=BBOX_CROP_OVERLAP,
                                                   assign_negative=False)
        return cropped_image, labels, bboxes, distort_bbox


def random_crop(img_data,labels,rects):

    with tf.Graph().as_default():
        img_in = tf.placeholder(tf.uint8)
        labels_in = tf.placeholder(tf.int32, shape=(None, ))
        boxes_in = tf.placeholder(tf.float32, shape=(None, 4))
        
        dst_image, dst_labels, dst_bboxes, distort_bbox = \
                distorted_bounding_box_crop(img_in, labels_in, boxes_in,
                                            min_object_covered=MIN_OBJECT_COVERED,
                                            aspect_ratio_range=CROP_RATIO_RANGE)
        
        img_float = tf.image.convert_image_dtype(img_in, dtype=tf.float32)
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(img_float,0),
                                                    tf.expand_dims(boxes_in,0))
        dst_image = tf.image.convert_image_dtype(dst_image, dtype=tf.float32)
        dist_image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(dst_image,0),
                                                    tf.expand_dims(dst_bboxes,0))
        
        
        img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        im_h,im_w,im_d = img_rgb.shape
        
        new_labels = []
        boxes = []
        for i in range(len(labels)):
            new_labels.append(int(labels[i]))
            x1 = box_thred(rects[i]["x1"] /float(im_w))
            x2 = box_thred(rects[i]["x2"] /float(im_w))
            y1 = box_thred(rects[i]["y1"] /float(im_h))
            y2 = box_thred(rects[i]["y2"] /float(im_h))
            boxes.append([y1,x1,y2,x2])
        
        config = tf.ConfigProto(
        device_count = {'GPU': 0}
    			)
        with tf.Session(config=config) as sess:
            dst_img,dst_bbo,dist_label = sess.run([dst_image,dst_bboxes,dst_labels],feed_dict={img_in: img_rgb, labels_in: new_labels, boxes_in:boxes })
            img_bgr =  (dst_img*255).astype(np.uint8)
            img_return = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            label_return =[]
            box_return =[]
            im_h,im_w,im_d = img_return.shape
            for i in range(dist_label.shape[0]):
                label_return.append(str(dist_label[i]))
                x1 = int(box_thred(dst_bbo[i][1]) *float(im_w))
                x2 = int(box_thred(dst_bbo[i][3]) *float(im_w))
                y1 = int(box_thred(dst_bbo[i][0]) *float(im_h))
                y2 = int(box_thred(dst_bbo[i][2]) *float(im_h))
                rect = {}
                rect["x1"] = x1
                rect["x2"] = x2
                rect["y1"] = y1
                rect["y2"] = y2
                box_return.append(rect)
            

            return img_return,label_return,box_return
