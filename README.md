https://www.kaggle.com/c/whale-categorization-playground

Categorizer for this Kaggle competition. To be used as part of my personal project website whale game 'computer' player.

Requires: Python 3+ as use type hints

Folder Structure:
- Data: where the Pascal XML files with annotations are located. conversion.py assumes that this is where the directory is
-

Resources Used:
- https://github.com/tensorflow/models/tree/master/research/object_detection
- https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
- https://github.com/priya-dwivedi/Deep-Learning/tree/master/tensorflow_toy_detector
- https://towardsdatascience.com/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95
- https://github.com/datitran/raccoon_dataset

General Steps:
- Download labelimg to manually annonate the images that had whale flukes in it
- labelimg outputs a type of file (XML) that holds all the necessary data but needs to get converted to a TFrecord format
- We do this by first translating them to CSV files (see `xml_to_csv.py` then taking these CSV files and converting
them to TFrecords (`conversion.py`)
