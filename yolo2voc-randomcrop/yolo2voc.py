import xml.etree.cElementTree as ET



def yolo2voc(img_id,img_w,img_h,img_d,labels,boxes,out_label_path):
    one_decimal = "{0:0.1f}"
    
    
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "Fashion Database"
    ET.SubElement(root, "filename").text = "%06d.jpg"%int(img_id)

    source = ET.SubElement(root, "source")
    owner = ET.SubElement(root, "owner")
    size = ET.SubElement(root, "size")

    ET.SubElement(source, "database").text = "The Gofind Database"
    ET.SubElement(source, "annotation").text = "Gofind"
    ET.SubElement(source, "image").text = "flickr"
    ET.SubElement(source, "flickrid").text = "341012865"

    ET.SubElement(owner, "flickrid").text = "Gofind"
    ET.SubElement(owner, "name").text = "Gofind"

    ET.SubElement(size, "width").text = str(img_w)
    ET.SubElement(size, "height").text = str(img_h)
    ET.SubElement(size, "depth").text = str(img_d)

    ET.SubElement(root, "segmented").text = "0"

    for i in range(len(labels)):
        _object = ET.SubElement(root, "object")

        ET.SubElement(_object, "name").text = labels[i]
        ET.SubElement(_object, "pose").text = "Left"
        ET.SubElement(_object, "truncated").text = "0"
        ET.SubElement(_object, "difficult").text = "0"
        bndbox = ET.SubElement(_object, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(boxes[i]['x1'])
        ET.SubElement(bndbox, "ymin").text = str(boxes[i]['y1'])
        ET.SubElement(bndbox, "xmax").text = str(boxes[i]['x2'])
        ET.SubElement(bndbox, "ymax").text = str(boxes[i]['y2'])


    tree = ET.ElementTree(root)
    tree.write(out_label_path)
