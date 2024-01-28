# TO-VisDrone-YOLO


## Use virtual environments (venv)
Install necessary dependencies:
```
git clone https://github.com/Piotr-Sicinski/TO-VisDrone-YOLO.git
cd TO-VisDrone-YOLO
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Example usage:

Flag ```--help``` can be used to learn more about possible arguments of scripts.
##

**vD2YOLO** - converter from VisDrone format annotations to YOLO labels,  ```--save-img``` to save jpg with applied bounding boxes
```
py vD2YOLO.py --source .\test-dir\
```
##

**bboxViever** - view a jpg with bounding boxes, also ```--save-img``` to save jpg
```
py bboxViever.py --source .\test-dir\ --imgn 0000001_02999_d_0000005 --save-img
```
##

**yolo_detection** - runs YOLO model on given image and calculates IoU, use ```--conf x``` to set detection treshold to x
```
py .\yolo_detection.py --source-image .\test-dir\images\0000001_02999_d_0000005.jpg --source-data .\test-dir\labels\0000001_02999_d_0000005.txt
```

##
Using ```python3``` instead of ```py``` may solve some problems.