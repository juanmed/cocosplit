import json
import argparse
import funcy
from sklearn.model_selection import train_test_split
import pathlib as pl
import os
import shutil


parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('train', type=str, help='Where to store COCO training annotations')
parser.add_argument('test', type=str, help='Where to store COCO test annotations')
parser.add_argument('-s', dest='split', type=float, required=True,
                    help="A percentage of a split; a number in (0, 1)")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')
parser.add_argument('-a', dest='traindir', type=str, help='Where to store the train split')
parser.add_argument('-b', dest='testdir', type=str, help='Where to store the test split')
parser.add_argument('-i', dest='inputdir', type=str, help='Directory where input images are')

args = parser.parse_args()

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def main(args):
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if not os.path.exists(args.traindir):
          pl.Path(args.traindir).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(args.testdir):
          pl.Path(args.testdir).mkdir(parents=True, exist_ok=True)  

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=args.split)
        save_coco(os.path.join(args.traindir,args.train), info, licenses, x, filter_annotations(annotations, x), categories)
        save_coco(os.path.join(args.testdir,args.test), info, licenses, y, filter_annotations(annotations, y), categories) 

        for file in x:
            shutil.copy(os.path.join(args.inputdir, file["file_name"]), args.traindir)
        for file in y:
            shutil.copy(os.path.join(args.inputdir, file["file_name"]), args.testdir)

        print("Saved {} entries in {} and {} in {}".format(len(x), args.train, len(y), args.test))


if __name__ == "__main__":
    main(args)