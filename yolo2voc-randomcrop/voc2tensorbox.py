import os
import sys
import json
import random

import numpy as np

import xml.etree.ElementTree as ET

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

outfile_train_name = './Yolo_train_data.json'
outfile_test_name = './Yolo_test_data.json'
one_decimal = "{0:0.1f}"
final_train_json = []
final_test_json = []

def box_thred(x):
    if x < 0:
        return 0
    elif  x > 1:
        return 1
    else:
        return x

def _process_image(directory, name):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    json_dict = {}
    # Read the image file.
    imgname = directory + DIRECTORY_IMAGES + name + '.jpg'
    json_dict["image_path"] = imgname
    json_dict["labels"] = []
    json_dict["rects"] = []
    #image_data = tf.gfile.FastGFile(imgname, 'r').read()

    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()
    
    # Find annotations.
    for obj in root.findall('object'):
        label = obj.find('name').text

        bbox = obj.find('bndbox')
        y1 = float(bbox.find('ymin').text)
        x1 = float(bbox.find('xmin').text)
        y2 = float(bbox.find('ymax').text)
        x2 = float(bbox.find('xmax').text) 
                      
        
        
        
        if (x1<x2) and (y1<y2):
            rect = {}
            rect["x1"] = float(one_decimal.format(x1))
            rect["x2"] = float(one_decimal.format(x2))
            rect["y1"] = float(one_decimal.format(y1))
            rect["y2"] = float(one_decimal.format(y2))
            json_dict["rects"].append(rect)
            json_dict["labels"].append(label)
    return json_dict


def run(dataset_dir, output_dir, name='voc_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames, and shuffling.
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Pascal VOC dataset!')


def main():

    root_directory = sys.argv[1]
    print root_directory

    for p in ["train","val"]:
        data_dir = root_directory + "/" + p + "/"
        path = os.path.join(data_dir, DIRECTORY_ANNOTATIONS)
        filenames = sorted(os.listdir(path))
        print p

        for _filename in filenames:
            img_name = _filename[:-4]

            if p == "train":
                final_train_json.append(_process_image(data_dir, img_name))
            else:
                final_test_json.append(_process_image(data_dir, img_name))
    
    outfile = open(outfile_train_name, 'w')
    json.dump(final_train_json, outfile, sort_keys = True, indent = 4)
    outfile.close()

    outfile = open(outfile_test_name, 'w')
    json.dump(final_test_json, outfile, sort_keys = True, indent = 4)
    outfile.close()

if __name__ == "__main__":
    main()