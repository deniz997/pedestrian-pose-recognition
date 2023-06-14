import sys
import cv2
import os
from sys import platform
import argparse


def import_op():
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/python/openpose/Release')
            os.add_dll_directory(dir_path + '/Release')
            os.add_dll_directory(dir_path + '/bin')
            global op
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/python/openpose/Release')
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            global op
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e


def get_keypoints_image(img_path: str):
    import_op()

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default=img_path,
                        help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Display Image
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("OpenPose 1.7.0 - Output Image", datum.cvOutputData)
    cv2.waitKey(0)


def set_params():
    '''
    Creates a parameter dictionary used by OpenPose
    :return: Parameter dictionary with default values
    '''

    dir_path = os.path.dirname(os.path.realpath(__file__))
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    # If GPU version is built, and multiple GPUs are available, set the ID here
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    params["model_folder"] = dir_path + "/models/"  # THIS IS RELATIVE TO PATH OF EXECUTABLE
    return params


def get_camera_stream_and_display():
    '''
    This is an example function to generate and display body keypoints on a camera input stream
    '''
    import_op()
    params = set_params()

    # Constructing OpenPose object allocates GPU memory
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Opening OpenCV stream
    stream = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:

        ret, img = stream.read()

        # Output keypoints and the image with the human skeleton blended on it
        # keypoints, output_image = opWrapper.start()
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        keypoints = datum.poseKeypoints
        output_image = datum.cvOutputData

        # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
        if len(keypoints) > 0:
            print('Human(s) Pose Estimated!')
            print(keypoints)
        else:
            print('No humans detected!')

        # Display the stream
        cv2.putText(output_image, 'OpenPose using Python-OpenCV', (20, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Human Pose Estimation', output_image)

        key = cv2.waitKey(1)

        # use 'q' to exit stream
        if key == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()


def get_camera_keypoints():
    '''
    Call this function to get the real time keypoints of the camera input. This function is
    very slow without a proper GPU installed.
    :return: Array of people and their body keypoints
    '''
    import_op()
    params = set_params()

    # Constructing OpenPose object allocates GPU memory
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Opening OpenCV stream
    stream = cv2.VideoCapture(0)

    ret, img = stream.read()

    # Output keypoints and the image with the human skeleton blended on it
    # keypoints, output_image = opWrapper.start()
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    keypoints = datum.poseKeypoints
    stream.release()

    return keypoints
