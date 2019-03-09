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


with tf.Graph().as_default():
    img_in = tf.placeholder(tf.uint8)
    labels_in = tf.placeholder(tf.int32, shape=(None, ))
    boxes_in = tf.placeholder(tf.float32, shape=(None, 4))
    
    dst_image, labels, dst_bboxes, distort_bbox = \
            distorted_bounding_box_crop(img_in, labels_in, boxes_in,
                                        min_object_covered=MIN_OBJECT_COVERED,
                                        aspect_ratio_range=CROP_RATIO_RANGE)
    


    img_float = tf.image.convert_image_dtype(img_in, dtype=tf.float32)
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(img_float,0),
                                                 tf.expand_dims(boxes_in,0))
    dst_image = tf.image.convert_image_dtype(dst_image, dtype=tf.float32)
    dist_image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(dst_image,0),
                                                 tf.expand_dims(dst_bboxes,0))
     
    image_path = sys.argv[1]
    label_path = sys.argv[2]
    img_data = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    im_h,im_w,im_d = img_data.shape
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
        x1 = ((xc - 0.5*w))
        x2 = ((xc + 0.5*w ))
        y1 = ((yc - 0.5*h))
        y2 = ((yc + 0.5*h))
        
        if (x1<x2) and (y1<y2):
            labels.append(int(_object[0]))
            boxes.append([y1,x1,y2,x2])
        else:
            pass
    
    with tf.Session() as sess:
        
            for i in xrange(5):
                image = cv2.imread(image_path)
                np_image,image_box,dst_bbo,dst_bbob = sess.run([dst_image,dist_image_with_box,dst_bboxes,distort_bbox],feed_dict={img_in: img_rgb, labels_in: labels, boxes_in:boxes })
                #print ori_bboxes
                #d_image,dist_image_box,np_label, np_bboxes = sess.run([dst_image,dist_image_with_box, labels, bboxes])
                #print np_bboxes
                #height, width, _ = np_image.shape
                #print height,width
                #print np_shape
                print dst_bbo
                print dst_bbob
                print labels
                #cv2.imshow("dis_image",np_image)
                #cv2.waitKey(0)
                
                plt.figure()
                
                plt.imshow(image_box[0])
                        
                plt.axis('off')
                plt.show()
                '''
                if len(ori_bboxes) > 0:
                        plt.figure()
                        plt.imshow(dist_image_box[0])
                        plt.title('%s, %d x %d' % (str(1), height, width))
                        plt.axis('off')
                        plt.show()
                '''
