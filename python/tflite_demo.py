import argparse
import time
import numpy as np
from PIL import Image
import tensorflow as tf

def tensor_image_dims(shape):
    """Extract width, height, and channels from a tensor shape array."""
    if len(shape) < 2:
        return None, None, None
    # Handle different tensor formats (NHWC, NCHW, etc.)
    if len(shape) == 4:  # NCHW or NHWC format
        if shape[1] == 1 or shape[1] == 3:  # NCHW
            return shape[3], shape[2], shape[1]
        else:  # NHWC
            return shape[2], shape[1], shape[3]
    elif len(shape) == 3:  # HWC format
        return shape[1], shape[0], shape[2]
    elif len(shape) == 2:  # HW format
        return shape[1], shape[0], 1
    return None, None, None

def print_tensor_info(tensor):
    """Print information about a tensor."""
    print(f"INFO:   Shape: {tensor.shape}")
    print(f"INFO:   Type: {tensor.dtype}")
    print(f"INFO:   Size: {tensor.nbytes} bytes")

def main():
    parser = argparse.ArgumentParser(description='Run TFLite model on an image')
    parser.add_argument('model', help='Path to the TFLite model file')
    parser.add_argument('input', help='Path to the input image')
    parser.add_argument('output', help='Path to save the output image')
    args = parser.parse_args()

    print(f"INFO: TFLite version: {tf.__version__}")

    # Load the model
    try:
        interpreter = tf.lite.Interpreter(model_path=args.model)
        
        # Set up QNN delegate
        try:
            # Load the QNN delegate library
            delegate_options = {
                'backend_type': 'htp'  # Using HTP backend as in the C version
            }
            delegate = tf.lite.experimental.load_delegate('libQnnTFLiteDelegate.so', delegate_options)
            interpreter = tf.lite.Interpreter(
                model_path=args.model,
                experimental_delegates=[delegate]
            )
            print("INFO: Loaded QNN delegate with HTP backend")
        except Exception as e:
            print(f"WARNING: Failed to load QNN delegate: {e}")
            print("INFO: Continuing without QNN delegate")
            interpreter = tf.lite.Interpreter(model_path=args.model)
        
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"ERROR: Failed to load model file '{args.model}': {e}")
        return 1

    print(f"INFO: Loaded model file '{args.model}'")

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1:
        print(f"ERROR: expected only 1 input tensor, got {len(input_details)}")
        return 1

    if len(output_details) != 1:
        print(f"ERROR: expected only 1 output tensor, got {len(output_details)}")
        return 1

    # Print input tensor info
    print("INFO: Input tensor:")
    print_tensor_info(interpreter.get_input_details()[0]['shape'])

    # Get input dimensions
    in_w, in_h, in_c = tensor_image_dims(interpreter.get_input_details()[0]['shape'])
    if not all([in_w, in_h, in_c]):
        print("ERROR: failed to extract image dimensions of input tensor.")
        return 1

    print(f"INFO: input tensor image dimensions: {in_w}x{in_h}, with {in_c} channels")

    # Print output tensor info
    print("INFO: Output tensor:")
    print_tensor_info(interpreter.get_output_details()[0]['shape'])

    # Get output dimensions
    out_w, out_h, out_c = tensor_image_dims(interpreter.get_output_details()[0]['shape'])
    if not all([out_w, out_h, out_c]):
        print("ERROR: failed to extract image dimensions of output tensor.")
        return 1

    print(f"INFO: output tensor image dimensions: {out_w}x{out_h}, with {out_c} channels")

    # Load and preprocess input image
    try:
        input_image = Image.open(args.input)
        if input_image.size != (in_w, in_h):
            print(f"ERROR: input image {args.input} does not match dimension of input tensor: "
                  f"expected {in_w}x{in_h}, got {input_image.size}")
            return 1

        # Get expected input dtype and shape
        expected_dtype = input_details[0]['dtype']
        expected_shape = input_details[0]['shape']
        expected_channels = in_c
        # Ensure image has the correct number of channels
        if expected_channels == 1:
            input_image = input_image.convert('L')
        elif expected_channels == 3:
            input_image = input_image.convert('RGB')
        # Convert image to numpy array with correct dtype
        if expected_dtype == np.uint8:
            input_data = np.array(input_image, dtype=np.uint8)
        else:
            input_data = np.array(input_image, dtype=np.float32)
            if input_data.max() > 1.0:
                input_data = input_data / 255.0
        # Add batch dimension if needed
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        # Reshape to expected input shape if needed
        if input_data.shape != tuple(expected_shape):
            input_data = input_data.reshape(tuple(expected_shape))
        print(f"INFO: loaded input image '{args.input}'")
    except Exception as e:
        print(f"ERROR: failed to open image '{args.input}': {e}")
        return 1

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    print("INFO: Invoking interpreter...")
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"INFO: Model execution time: {elapsed_time:.2f} ms")

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Remove batch dimension if present
    if len(output_data.shape) == 4:
        output_data = output_data[0]

    # Convert to uint8 if needed
    if output_data.max() <= 1.0:
        output_data = (output_data * 255).astype(np.uint8)

    # Save output image
    try:
        output_image = Image.fromarray(output_data)
        output_image.save(args.output)
        print(f"INFO: wrote output image to '{args.output}'")
    except Exception as e:
        print(f"ERROR: Failed to write output to '{args.output}': {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())