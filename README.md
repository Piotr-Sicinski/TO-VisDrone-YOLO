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

## bboxViever example usage:
```
cd bboxViever
```

Flag ```--help``` can be used to learn more about possible arguments of ```py vD2YOLO.py``` and  ```py bboxViever.py```

vD2YOLO - converter from VisDrone format annotations to YOLO labels,  ```--save-img``` to save jpg with applied bounding boxes
bboxViever - view a jpg with bounding boxes, also ```--save-img``` to save jpg


```
py vD2YOLO.py --source .\test-dir\
py bboxViever.py --source .\test-dir\ --imgn 0000001_02999_d_0000005 --save-img
```


Using ```python3``` instead of ```py``` may solve some problems.