# └── working directory
#      └── 'source'
#               └── annotations
#               └── images
#               └── labels (will be created)

from PIL import Image
from pathlib import Path
import argparse
import bboxViever

DIR = Path(__file__).absolute().parent

IMAGE_FOLDER = "images"
LABEL_FOLDER = "labels"
ANNOTATIONS_FOLDER = "annotations"

REMOVE_IGNORED = True


def convert_annotation(img_size, bbox):
    # Convert VisDrone bounding box to YOLO bounding box in xywh pattern
    width_div = 1.0 / img_size[0]
    height_div = 1.0 / img_size[1]
    return [(bbox[0] + bbox[2] / 2) * width_div, (bbox[1] + bbox[3] / 2) * height_div, bbox[2] * width_div, bbox[3] * height_div]


def main(source, save_img):
    dataFolder = Path(source)

    label_dir = dataFolder / LABEL_FOLDER
    image_dir = dataFolder / IMAGE_FOLDER
    annotations_dir = dataFolder / ANNOTATIONS_FOLDER

    if not label_dir.is_dir():
        label_dir.mkdir()

    # Read all filenames in the original annotations directory and add the names to a list
    ann_files = [file for file in annotations_dir.iterdir()
                 if file.is_file()]

    for file in ann_files:

        with open(file, 'r', encoding='utf8') as f:
            img = Image.open(image_dir / (file.stem + '.jpg'))

            for line in f:
                # Separate individual elements in the annotations separated by ','
                data = line.strip().split(',')
                # Assume YOLO classes in the range 0-9 with 0-pedestrian and 9-motor
                class_label = int(data[5]) - 1

                if REMOVE_IGNORED:
                    # If ignored annotations should be removed, check whether current annotation is considered
                    considered = data[4]
                else:
                    considered = 1  # If ignored annotations are not to be removed, consider all annotations

                # Check for valid classes
                if ((considered != str(0)) and (class_label >= 0) and (class_label <= 9)):
                    bounding_box_visdrone = [float(x) for x in data[:4]]
                    yolo_bounding_box = convert_annotation(
                        img.size, bounding_box_visdrone)
                    # Create the annotation string to be written
                    bounding_box_string = " ".join(
                        [str(x) for x in yolo_bounding_box])

                    with open(label_dir / file.name, 'a+', encoding="utf-8") as output_file:
                        # Write the converted annotation with class in YOLO format
                        output_file.write(
                            f"{class_label} {bounding_box_string}\n")
        if save_img:
            bboxViever.viewer(source, file.stem, save_img, False)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images',
                        help='file/dir')
    parser.add_argument('--save-img', action='store_true',
                        help='save results to \'source\'/output/*.jpg')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
