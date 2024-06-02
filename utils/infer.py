import onnxruntime as rt
import numpy as np
import struct 

def load_mnist_image(filename):
    """Loads a single MNIST image from a .ubyte file."""
    with open(filename, 'rb') as f:
        # Read image data as bytes
        image_data = f.read(28 * 28) 
        # Convert to numpy array, reshape, and normalize
        image = np.frombuffer(image_data, dtype=np.uint8).reshape(1, 1, 28, 28).astype(np.float32)
    return image

# Load the ONNX model
sess = rt.InferenceSession("../models/mnist_ffn.onnx")


# Get input and output names
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print("input name:", input_name)
print("output name:", output_name)


# Load and preprocess the MNIST image (now as float32)
image = load_mnist_image("../inputs/image_0.ubyte")

# Run inference
result = sess.run([output_name], {input_name: image})

# Get predicted class
predicted_class = np.argmax(result)

print(result)

print("Predicted class:", predicted_class)