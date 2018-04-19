https://www.kaggle.com/c/whale-categorization-playground

Categorizer for this Kaggle competition. To be used as part of my personal project website whale game 'computer' player.

Requires: Python 3+ as use type hints

Folder Structure:
- Data: where the Pascal XML files with annotations are located. conversion.py assumes that this is where the directory is
-

Major Resources Used (other than documentation from pandas, tensorflow and numpy):
- https://github.com/tensorflow/models/tree/master/research/object_detection
- https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
- https://github.com/priya-dwivedi/Deep-Learning/tree/master/tensorflow_toy_detector
- https://towardsdatascience.com/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95
- https://github.com/datitran/raccoon_dataset


General Steps:
- Download labelimg to manually annonate the images that had whale flukes in it
- labelimg outputs a type of file (XML) that holds all the necessary data but needs to get converted to a TFrecord format (manual step using LabelImg)
- We do this by first translating them to CSV files / dataframe objects(see `xml_to_csv.py` then taking these and converting
them to TFrecords (`conversion.py`). Don't have to save to CSV file. Only do it if you want a nice readable CSV of all
the images and their corresponding details. (This work is done in conversion.py and then user interacts with it through
run_converter.py via command line to ensure that the relevant images are converted to TFRecord). NOTE: this method creates
a subdir `conversion.output_path` within this home directory where it puts the TFRecord files and any CSV files
- Then train the model using: https://github.com/tensorflow/models/blob/master/research/object_detection/train.py#L17
    -Note: git cloned https://github.com/tensorflow/models
    -Note: I cloned the repo to some other directory not within this 'Whales' directory as I wanted it to be easily
    accessible to other projects. And thus in order to get this to work on your machine as described you'd have to clone
    the same repo onto your computer and follow the installation steps accordingly
    -Note: then followed installation instructions - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
    -Note: To run the protobuf compilation command: protoc object_detection/protos/*.proto --python_out=. found here:
     https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
     I had to make sure my conda environment that I installed all of the necessary dependencies was activated before being
     able to successfully compile protobuf
    -Note: Running this: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#add-libraries-to-pythonpath
    is required before the test installation step will work. Or else if you don't add it when you open a new terminal the
    test step will give you an error
Decided not to wrap it with my own command line function. user then runs train.py in the terminal with the appriorate flags see
below:


ex.) For the 'simple usage' do not have a good example of the 2nd usage yet as don't understand the last input of 'input_config_path' and
where that comes from as see 'model_config' and 'train_config' sections within the pipeline config file (we use faster_rcnn_resnet101_coco.config)
but don't see 'input_config_path'

python train.py --logtostderr --train_dir=path/to/train_dir --pipeline_config_path=pipeline_config.pbtxt

Explaining the above.
--train_dir value is the directory where all the training output for the model will go, the actual
model graph file and any tensorboard related files. So for us we have 'training/'. This is the directory that we point
to when we run tensorboard and want to see the training progress - (`tensorboard --logdir=training/`)
-- pipeline_config_path value is the path to the 'TrainEvalPipelineConfig configuration file' that we use to config the
file will whatever preliminary settings we want. This file also tells the model how many labels it needs to classify,
where to save model checkpoints, and where to find the data to be used for testing and validation ->i.e note that
 when calling `train.py` we do not tell it where the model can find the training and testing data that's because
 that info is in the config file itself (in this project I use the faster_rcnn_resnet101_coco.config under the
 faster_rcnn_resnet101_coco_2018_01_28 directory)
-- Note: For me the train_dir path had to be directed to the directory within 'Whales' which was in another directoy
as well as the pipeline path which had to point to the faster_rcnn_resnet101_coco_2018_01_28 directory
