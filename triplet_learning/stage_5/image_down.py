import io
import os
import json
from PIL import Image
from io import BytesIO
import urllib2 as urllib



file_name = "./query_and_triplets.txt"
OUT_FILE = "./triplet_fine_new.json"
Data_Json = []

index = 0
json_dict = {}

def download_img(link_add, name, type_in,index):
    fd = urllib.urlopen(str(link_add))
    image_file = io.BytesIO(fd.read())
    im = Image.open(image_file)
    save_name = name + "_" + type_in + "_"+ index + ".jpg"
    im.save("./image/"+save_name,format="JPEG")
    return save_name


with open(file_name) as f:

    for line in f:
        print index
        if (index%4 == 0):
            name = line[:-1]
            json_dict["name"] = name

        if (index%4 == 1):
            link_add = line[:-1]
            print name
            image_name = download_img(link_add,name,"anchor",str(index//4))
            json_dict["anchor"] = os.getcwd() +"/image/"+ image_name
            json_dict["anchor_class"] = 0

        if (index%4 == 2):
            link_add = line[:-1]
            print name
            image_name = download_img(link_add,name,"positive",str(index//4))
            json_dict["positive"] = os.getcwd() +"/image/"+ image_name
            json_dict["positive_class"] = 1

        if (index%4 == 3):
            link_add = line[:-1]
            print name
            image_name = download_img(link_add,name,"negative",str(index//4))
            json_dict["negative"] = os.getcwd() +"/image/"+ image_name
            json_dict["negative_class"] = 2
            Data_Json.append(json_dict)
            json_dict = {}

        index += 1

        if index > 10:
            break

outfile = open(OUT_FILE, 'wb')
json.dump(Data_Json, outfile,sort_keys = True, indent = 4)
outfile.close()
