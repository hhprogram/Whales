import argparse
import conversion
# file that is the entry point to the training the WhaleClassifier. Run this file with the apprioriate tags in order
# to (1) create the necessary record files from a directory of XML files

def train_test_split(input):
    try:
        train, test = map(int, input.split(","))
        if train + test != 100 and train + test != 1:
            raise argparse.ArgumentTypeError("Train-Test split must sum to 100%")
        return train, test
    except:
        raise argparse.ArgumentTypeError("Train-Test split must be in <train %>, <test %> format, summing to 100%")

def run_converter():
    parser = argparse.ArgumentParser()
    # note: any argument that starts with either '-' or a '--' is an optional argument in the command line
    parser.add_argument("--xml_dir", help="Relative path to directory of xmls to be converted to TFRecord if applicable")
    parser.add_argument("--jpg_path", help="Relative path to directory of where the original jpg images are")
    # ---optional arguments below
    parser.add_argument("--csv", help="Leave blank if don't want a CSV of all the images and their details, put relative path to directory if do want CSV file")
    # see: https://docs.python.org/3/library/argparse.html#type, for how I'm using the type argument
    # NOTE: I know now I could just have argument be train % and take the remaining to be validation but leaving this
    # in for pedagogical purposes or learning more about add_argument. the type argument and my train_test_split
    # callable ensures that the split variable in the parse_args() object will be a tuple with normal tuple index
    # referencing
    parser.add_argument("--split",
                        help="Split between % images used in training and validation If not given default 80/20.format= train %, test %",
                        type=train_test_split)
    parser.add_argument("--trainRecordName",
                        help="If you want to customize the filename for train.record to be used to train model. Default to train.record")
    parser.add_argument("--testRecordName",
                        help="If you want to customize the filename for test.record to be used to train model. Default to test.record")
    # parser.add_argument('--foo', help='foo help')
    # call parse_args on the parser object to get the namespace object that holds all the args. (i.e can then access the
    # argument values using dot notation. Where you would do args.foo to get the --foo argument. Ie the text following
    # the --foo flag. ex.) python run_converter.py --foo Hello --> Namespace(foo="Hello") --> args.foo -> "Hello"
    args = parser.parse_args()
    # Note: if user doesn't fill in the argument then it is a None type.
    if args.xml_dir == None or args.jpg_path == None:
        raise argparse.ArgumentTypeError("xml_dir or jpg_path is left blank, must provide relative path")
    train_percent = args.split[0] if args.split != None else .8
    # actually don't use the below but keeping it to learn how argparser works
    test_percent = args.split[1] if args.split != None else .2
    train_name = args.trainRecordName if args.trainRecordName else "train"
    test_name = args.testRecordName if args.testRecordName else "test"
    csv_bool = True if args.csv else False
    conversion.run_conversion(xml_dir=args.xml_dir,
                              jpg_dir=args.jpg_path,
                              train_percent=train_percent,
                              trainName=train_name,
                              testName=test_name,
                              csv=csv_bool)

if __name__ == "__main__":
    run_converter()