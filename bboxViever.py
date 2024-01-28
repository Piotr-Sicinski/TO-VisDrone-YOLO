# └── working directory
#      └── 'source'
#               └── images
#               └── labels
#               └── output (will be created)

from PIL import Image, ImageDraw
from pathlib import Path
import argparse

DIR = Path(__file__).absolute().parent
IMAGE_FOLDER = "images"
LABEL_FOLDER = "labels"
OUTPUT_FOLDER = "output"


# To convert annotations from YOLO format to absolute corner points of a rectangle
def yolo_to_bounding_box(image_class, b_box, w, h):
    # x_center, y_center, width, heigth
    half_width = (b_box[2] * w) / 2
    half_height = (b_box[3] * h) / 2
    x_min = int((b_box[0] * w) - half_width)
    y_min = int((b_box[1] * h) - half_height)
    x_max = int((b_box[0] * w) + half_width)
    y_max = int((b_box[1] * h) + half_height)
    return [image_class, x_min, y_min, x_max, y_max]


# To draw bounding boxes on the image
def draw_boxes(img, b_boxes):
    draw = ImageDraw.Draw(img)

    # To assign color based on class label
    color_list = ["red", "green", "blue", "yellow", "purple",
                  "orange", "pink", "teal", "magenta", "turquoise"]

    for b_box in b_boxes:
        draw.rectangle(b_box[1:], outline=color_list[int(b_box[0])], width=1)


def viewer(source, imgn, save_img, show_img=True):
    dataFolder = Path(source)
    label_filepath = dataFolder / LABEL_FOLDER / (imgn + '.txt')

    img = Image.open(dataFolder / IMAGE_FOLDER / (imgn + '.jpg'))
    bounding_boxes = []

    with open(label_filepath, 'r', encoding='utf8') as file:
        for line in file:
            data = line.strip().split(' ')
            image_class = data[0]
            bounding_box = [float(val) for val in data[1:]]
            bounding_boxes.append(yolo_to_bounding_box(
                image_class, bounding_box, img.size[0], img.size[1]))

    draw_boxes(img, bounding_boxes)
    if show_img:
        img.show()
    if save_img:
        output_path = dataFolder / OUTPUT_FOLDER
        if not output_path.is_dir():
            output_path.mkdir()
        img.save(output_path / (imgn + '_output.jpg'))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images',
                        help='path to folder containing data to analize')
    parser.add_argument('--imgn', required=True, type=str,
                        help='image name')
    parser.add_argument('--save-img', action='store_true',
                        help='save results to \'source\'/output/*.jpg')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    viewer(**vars(opt))
