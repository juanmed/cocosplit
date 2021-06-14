import skimage 
import skimage.io as io
import math
from itertools import chain
import numpy as np
import json
import os

#https://stackoverflow.com/questions/61210420/converting-the-annotations-to-coco-format-from-mask-rcnn-dataset-format


IMAGE_DIR = "C:\\Users\\le\\Documents\\projects\\dataset\\skku_DeepingSource\\val" # The directory where your images are stored.
VGG_DIR = "C:\\Users\\le\\Documents\\projects\\dataset\\skku_DeepingSource\\val\\via_region_data.json" # Either the directory or the path to your coco.json file given.
COCO_DIR = "C:\\Users\\le\\Documents\\projects\\dataset\\coco_skku_DeepingSource\\" # The directory in which the YOLO.txt files will be stored.

IN_FORMAT = "VGG" # The format of your annotations in this case COCO.
OUT_FORMAT = "COCO" # The output format you want in this case YOLO.

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def vgg_to_coco(dataset_dir, vgg_path: str, outfile: str=None, class_keyword: str = "label"):
    with open(vgg_path) as f:
        vgg = json.load(f)

    classes = {'box', 'pouch', 'icebox', 'sack'}

    images_ids_dict = {}
    images_info = []
    for i,v in enumerate(vgg.values()):

        images_ids_dict[v["filename"]] = i
        image_path = os.path.join(dataset_dir, v['filename'])
        image = io.imread(image_path)
        height, width = image.shape[:2]  
        images_info.append({"file_name": v["filename"], "id": i, "width": width, "height": height})

    print("Read VGG JSON...", len(images_info), len(images_ids_dict))

    #classes = {class_keyword} | {r["region_attributes"][class_keyword] for v in vgg.values() for r in v["regions"]
    #                         if class_keyword in r["region_attributes"]}
    #print(classes)
    classes = {'box', 'pouch', 'icebox', 'sack'}
    category_ids_dict = {c: i for i, c in enumerate(classes, 1)}
    categories = [{"supercategory": class_keyword, "id": v, "name": k} for k, v in category_ids_dict.items()]
    annotations = []
    suffix_zeros = math.ceil(math.log10(len(vgg)))
    for i, v in enumerate(vgg.values()):
        for j, r in enumerate(v["regions"]):
            if class_keyword in r["region_attributes"]:
                x, y = r["shape_attributes"]["all_points_x"], r["shape_attributes"]["all_points_y"]
                annotations.append({
                    "segmentation": [list(chain.from_iterable(zip(x, y)))],
                    "area": PolyArea(x, y),
                    "bbox": [min(x), min(y), max(x)-min(x), max(y)-min(y)],
                    "image_id": images_ids_dict[v["filename"]],
                    "category_id": category_ids_dict[r["region_attributes"][class_keyword]],
                    "id": int(f"{i:0>{suffix_zeros}}{j:0>{suffix_zeros}}"),
                    "iscrowd": 0
                    })

    coco = {
        "images": images_info,
        "categories": categories,
        "annotations": annotations
        }
    if outfile is None:
        outfile = vgg_path.replace(".json", "_coco.json")
    with open(outfile, "w") as f:
        json.dump(coco, f)

vgg_to_coco(IMAGE_DIR, VGG_DIR,outfile="trainval.json",class_keyword="type")
