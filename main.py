"""From here: https://www.tensorflow.org/lite/models/convert/metadata_writer_tutorial#object_detectors"""

import sys
import tensorflow as tf
from tensorflow.lite.support.metadata import metadata as tf_metadata
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils


def main(model_input_path, label_input_path, model_output_path) -> None:
    """Creates the metadata for an object detection model."""      
    
    input_norm = tf_metadata.NormalizationOptions(mean=[0.5], std=[0.5])
    model_metadata = tf_metadata.create_metadata()
    model_metadata.input_normalization.append(input_norm)

    model_metadata_buffer = model_metadata.serialize_to_buffer()
    open("models/birdbot.tflite", "wb").write(model_metadata_buffer)
    print("Done")


if __name__ == "__main__":
    #if len(sys.argv) != 4:
     #   print("Usage: python main.py <model_input_path> <label_input_path> <model_output_path>")
      #  sys.exit(1)
    
    # model_input_path = sys.argv[1]
    # label_input_path = sys.argv[2]
    #model_output_path = sys.argv[3]

#    print("model_input_path: ", model_input_path)
 #   print("label_input_path: ", label_input_path)
  #  print("model_output_path: ", model_output_path)
    main()
    # main(model_input_path, label_input_path, model_output_path)


