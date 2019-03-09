# yolo2voc-randomcrop

Hi, this is a project to transform **YOLO** training data files structure to **Pascal VOC** files structure.  Besides, the **distorted_bounding_box_crop** (introduced in SSD implemention [[arxiv]](https://arxiv.org/abs/1512.02325)) is also incorporated in the generating process. Thanks to **balancap** for implemention of SSD in tensorflow form [(Project)](https://github.com/balancap/SSD-Tensorflow). The project is still in process.

# Dependencies
* **python 2.7**
* **opencv 2/3**
* **tensorflow >=1.0**
 
# Implemention 
1. Prepare your training data in YOLO files structure
> 	JPEGImages: Training images
>
> 	Labels: Traing labels in format ( label, x_c, y_c, w, h )

2. Run Code
>   **usage**: dataset_build.py [-h] [-s IMAGE_DIR] [-l LABEL_DIR] [-o OUTPUT_ROOT][-val VAL_RATIO] [-d_p DIT_PRO] [-debug DEBUG]

>  optional arguments:
  >> **-h**, --help      show this help message and exit
  >> 
  >> **-s**  IMAGE_DIR    Directory of training images
  >> 
  >> **-l**  LABEL_DIR    Directory of training labels
  >> 
  >> **-o**  OUTPUT_ROOT  root directory of output training data
  >> 
  >> **-val** VAL_RATIO  train_validation split ratio
  >> 
  >> **-d_p** DIT_PRO    image files ratio for distorted_bounding_box_crop
  >> 
  >> **-debug DEBUG**    Debug flag, set True if you need to see the output image with bouding boxes
