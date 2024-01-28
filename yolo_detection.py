from ultralytics import YOLO
from pathlib import Path
import argparse
from PIL import Image

DIR = Path(__file__).absolute().parent

PREDICTIONS_FOLDER = DIR / "predictions"

yolo_class_dict = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    # 4 : "airplane",
    5: "bus",
    # 6 : "train",
    7: "truck",
    # 8 : "boat"
}
visdrone_class_dict = {
    0: "person",
    1: "person",  # people
    2: "bicycle",
    3: "car",
    5: "truck",
    8: "bus",
    9: "motorcycle"  # motor
}


class Box:
    def __init__(self, type, x, y, width, height):
        self.type = type
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"{self.type} {self.x} {self.y} {self.width} {self.height}"


def yolo(image_path, save_path, confidence):
    if not PREDICTIONS_FOLDER.is_dir():
        PREDICTIONS_FOLDER.mkdir()

    model = YOLO(f"yolov8n.pt")
    results = model.predict(source=image_path, conf=confidence/100)

    with open(PREDICTIONS_FOLDER / save_path, 'w') as f:
        for r in results:
            for b in r.boxes:
                f.write(f"{int(b.cls[0])} {b.xywhn[0][0]} {b.xywhn[0][1]} {b.xywhn[0][2]} {b.xywhn[0][3]}\n")

# calculating probability that predicted_box represent original box


def bb_intersection_over_union(original_box, predicted_box):

    xA = max(original_box.x, predicted_box.x)
    yA = max(original_box.y, predicted_box.y)
    xB = min(original_box.width, predicted_box.width)
    yB = min(original_box.height, predicted_box.height)

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # rectangles
    boxAArea = (original_box.width - original_box.x + 1) * (original_box.height - original_box.y + 1)
    boxBArea = (predicted_box.width - predicted_box.x + 1) * (predicted_box.height - predicted_box.y + 1)

    return interArea / float(boxAArea + boxBArea - interArea)


def load_bouding_boxes(yolo_path, visdrone_path):
    original_boxes = []
    predicted_boxes = []
    # serialization results from YOLO
    with open(PREDICTIONS_FOLDER / yolo_path, 'r') as calculated_file:
        for row in calculated_file:
            fields = [calculated_values for calculated_values in row.strip().split()]
            if int(fields[0]) in yolo_class_dict:
                predicted_boxes.append(Box(yolo_class_dict[int(fields[0])], fields[1], fields[2], fields[3], fields[4]))

    with open(visdrone_path, 'r') as original_file:
        for row in original_file:
            fields = [annotations_value for annotations_value in row.strip().split()]
            if int(fields[0]) in visdrone_class_dict:
                original_boxes.append(Box(visdrone_class_dict[int(fields[0])],
                                      fields[1], fields[2], fields[3], fields[4]))
            # add_object(fields, "visdrone")

        return original_boxes, predicted_boxes


def calculating_iou(original_boxes, predicted_boxes):
    # counting
    detected_people = 0
    detected_vehicles = 0
    sum_predictions = 0
    for predicted_box in predicted_boxes:

        if predicted_box.type == "person":
            detected_people += 1
        else:
            detected_vehicles += 1

        # print(f"PREDICT: {predicted_box}")
        the_highest_prediction = 0
        for original_box in original_boxes:
            if (predicted_box.type == original_box.type):
                rate_of_prediction = bb_intersection_over_union(original_box, predicted_box)
                if (rate_of_prediction > the_highest_prediction):
                    the_highest_prediction = rate_of_prediction
                # print(f"\tORIGIN {original_box} \t {round(bb_intersection_over_union(original_box, predicted_box), 4)}")
        sum_predictions += the_highest_prediction

    count_all_people = sum(obj.type == 'person' for obj in original_boxes)
    count_all_vehicles = sum(obj.type != 'person' for obj in original_boxes)

    iou = sum_predictions / len(predicted_boxes)
    iou_for_all_objects = sum_predictions / len(original_boxes)

    print(f"Has been detected {len(predicted_boxes)}/{len(original_boxes)} objects")
    print(f" - people: {detected_people}/{count_all_people}")
    print(f" - vehicles: {detected_vehicles}/{count_all_vehicles}")
    print(f"Average IoU for detected objects:{iou:.2f}")
    print(f"Average IoU for all objects:{iou_for_all_objects:.2f}")


def main(source_image, source_data, conf):

    prediction_file = f"prediction{source_image[-8:-4]}.txt"

    yolo(source_image, prediction_file, conf)

    original_boxes, predicted_boxes = load_bouding_boxes(prediction_file, source_data)

    calculating_iou(original_boxes, predicted_boxes)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-image', type=str,
                        help='relative path to image')  # JPG jako pierwsze
    parser.add_argument('--source-data', type=str,
                        help='relative path to image data txt')  # txt czyli labels, tutaj sa dane oryginalne bouding boxy tak aby program mogl policzyc IoU
    parser.add_argument('--conf', type=int, default=25,
                        help='confidence threshold')  # prog detekcji, tu ustawiony na 25
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
