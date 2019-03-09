"""This script is the transfer label resutls to train results
"""

import json

sku_imgs_dir = '/home/shangliy/Projects/FashionRecom/data_pro/tool_ui/static/photos/'

input_json_path = 'data/recom_res.json'
with open(input_json_path) as f:
    recom_res = json.load(f)

recom_res_nodup = {}
for rec in recom_res:
    recom_res_nodup[rec] = {}
    good_ = set()
    medium_ = set()
    bad_ = set()
    for skuid in recom_res[rec]['good']:
        good_.add(skuid)

    for skuid in recom_res[rec]['medium']:
        medium_.add(skuid)

    for skuid in recom_res[rec]['bad']:
        bad_.add(skuid)

    recom_res_nodup[rec]['good'] = list(good_)
    recom_res_nodup[rec]['medium'] = list(medium_)
    recom_res_nodup[rec]['bad'] = list(bad_)

triplet_list = []
for rec in recom_res_nodup:
    tri_set = {}
    tri_set['a'] = sku_imgs_dir + "%s.jpg"%(rec)
    if len(recom_res_nodup[rec]['good']) > 0 and len(recom_res_nodup[rec]['medium']) >0 :
        for i in range(len(recom_res_nodup[rec]['good'])):
            for j in range(len(recom_res_nodup[rec]['medium'])):
                tri_set['p'] = sku_imgs_dir + "%s.jpg"%(recom_res_nodup[rec]['good'][i])
                tri_set['n'] = sku_imgs_dir + "%s.jpg"%(recom_res_nodup[rec]['medium'][j])
                triplet_list.append(tri_set)
    if len(recom_res_nodup[rec]['good']) > 0 and len(recom_res_nodup[rec]['bad']) >0 :
        for i in range(len(recom_res_nodup[rec]['good'])):
            for j in range(len(recom_res_nodup[rec]['bad'])):
                tri_set['p'] = sku_imgs_dir + "%s.jpg"%(recom_res_nodup[rec]['good'][i])
                tri_set['n'] = sku_imgs_dir + "%s.jpg"%(recom_res_nodup[rec]['bad'][j])
                triplet_list.append(tri_set)
    if len(recom_res_nodup[rec]['good']) == 0 and len(recom_res_nodup[rec]['medium']) > 0 and len(recom_res_nodup[rec]['bad']) >0 :
        for i in range(len(recom_res_nodup[rec]['medium'])):
            for j in range(len(recom_res_nodup[rec]['bad'])):
                tri_set['p'] = sku_imgs_dir + "%s.jpg"%(recom_res_nodup[rec]['medium'][i])
                tri_set['n'] = sku_imgs_dir + "%s.jpg"%(recom_res_nodup[rec]['bad'][j])
                triplet_list.append(tri_set)

with open('data/triplet_data_list.json', 'w') as f:
    json.dump(triplet_list, f, indent=4, ensure_ascii=False)
