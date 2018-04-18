# file that holds all methods to convert inputs into necessary formats.
# we start from regular XML formatted files that are outputted from a program like labelimg (used to annotate) images
# so we can train our network to detect a custom image
# then we take the xml_to_csv function taken from:
# https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py
# Then we feed this CSV file into the create_tf_examples function that is based on the outline given here:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Union
import io

import tensorflow as tf

from object_detection import dataset_util #note changed this slightly as didn't have a utils folder just named it object_detection


csv_column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
path_column = "path"
# NOTE: should replace this with a dynamic function that can read the label map pbtxt and then create a dictionary
# to map the class text to the class id
label_map = {"fluke": 1}

def xml_to_df(path: str) -> pd.DataFrame:
    # taken from: https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py
    """takes a path to a directory that holds XML files output in PASCAL annotated format converts to dataframe object
    :returns a dataframe object that can be easily turned into a CSV or just passed on to another function to
    convert each row to a TFrecord"""
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    xml_df = pd.DataFrame(xml_list, columns=csv_column_names)
    # added this part on the bottom so that the dataframe object has the complete path as well for each file
    xml_df['path_column'] = os.path.join(path, xml_df[csv_column_names[0]])
    return xml_df


def df_to_csv(df: pd.DataFrame , output_path: str) -> None:
    df.to_csv(output_path, index=None)
    print('Successfully converted xml to csv.')


def create_tf_example(input: pd.Series) -> tf.train.Example:
    #   outputs one tf record per call. Therefore, to convert all relevant images to tf records we need to loop and
    # call this method on each EXAMPLE
    """example: the input that holds all necessary info to convert it to a tf record. Not actual JPG file but
    some object that holds all the info about that jpg
    INPUT: is a pandas series
    :returns a tf.train.Example which i think is then transformed into a string and that string is used as input to be
    written by a TFRecord writer into a file, and that file is in TFRecord format"""
    height = input[csv_column_names[2]] # Image height
    width = input[csv_column_names[1]] # Image width
    filename = input[csv_column_names[0]] # Filename of the image. Empty if image is not from file. Note not the whole path. only the filename
    # NOTE: gfile is mostly just a wrappe for Python's filesystem with open API. But can handle opening files
    # that are not local (like on google storage and HDFS as well)
    # see: https://stackoverflow.com/questions/42256938/what-does-tf-gfile-do-in-tensorflow?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    with tf.gfile.GFile(input[path_column], 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_image_data = io.BytesIO(encoded_jpg) # Encoded image bytes
    image_format = b'jpg' # b'jpeg' or b'png' - assume jpg files

    # below lists just have one element in them. see example:
    # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
    xmins = [input[csv_column_names[4]]] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [input[csv_column_names[6]]] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [input[csv_column_names[5]]] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [input[csv_column_names[7]]] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [input[csv_column_names[3]]] # List of string class name of bounding box (1 per box)
    classes = [label_map[classes_text]] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def write_TFRecord_file(input: Union[str, pd.DataFrame]) -> None:
    global label_map
    """INPUT can be a path to a CSV file that holds the xml file details. Or it's a dataframe object that holds the details"""
    if isinstance(input, str)
        # read in the CSV and turn it into a dataframe object
        images = pd.read_csv(input)
        writer = tf.python_io.TFRecordWriter(input)
    else:
        # if not we assume it's a dataframe object and then just set it to images
        images = input
    label_map =
    # loop through the dataframe object by each row. ROW is a 1 row pd.Series object that you can treat just like any
    # other dataframe object
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iterrows.html
    # it returns a tuple the first object in tuple is just the index of that row.
    for _, row in images.iterrows():
        tf_example = create_tf_example(row)
        writer.write(tf_example.SerializeToString())

    writer.close()