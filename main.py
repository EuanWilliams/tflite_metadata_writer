"""From here: https://www.tensorflow.org/lite/models/convert/metadata_writer_tutorial#object_detectors"""

import sys
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils


def main(model_input_path, label_input_path, model_output_path) -> None:
    """Creates the metadata for an object detection model."""      
    
    ObjectDetectorWriter = object_detector.MetadataWriter
    # Normalization parameters is required when reprocessing the image. It is
    # optional if the image pixel values are in range of [0, 255] and the input
    # tensor is quantized to uint8. See the introduction for normalization and
    # quantization parameters below for more details.
    # https://www.tensorflow.org/lite/models/convert/metadata#normalization_and_quantization_parameters)
    _INPUT_NORM_MEAN = 0
    _INPUT_NORM_STD = 255

    # Create the metadata writer.
    writer = ObjectDetectorWriter.create_for_inference(
        writer_utils.load_file(model_input_path), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD],
        [label_input_path])

    # Verify the metadata generated by metadata writer.
    print(writer.get_metadata_json())

    # Populate the metadata into the model.
    writer_utils.save_file(writer.populate(), model_output_path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <model_input_path> <label_input_path> <model_output_path>")
        sys.exit(1)
    
    model_input_path = sys.argv[1]
    label_input_path = sys.argv[2]
    model_output_path = sys.argv[3]

    print("model_input_path: ", model_input_path)
    print("label_input_path: ", label_input_path)
    print("model_output_path: ", model_output_path)

    main(model_input_path, label_input_path, model_output_path)

