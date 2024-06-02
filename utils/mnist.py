import struct
from PIL import Image

def extract_mnist_image(input_file, output_file, image_index):
    with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        # Read the MNIST header (we'll skip it for now)
        struct.unpack('>IIII', f_in.read(16))

        # Seek to the start of the desired image
        image_offset = 16 + image_index * 784
        f_in.seek(image_offset)

        # Read the image data (784 bytes)
        image_data = f_in.read(784)

        # Write the image data to the output file
        f_out.write(image_data)


def view_mnist_image(filename):
    with open(filename, 'rb') as f:
        image_data = f.read()

    image = Image.frombytes('L', (28, 28), image_data)  # L means grayscale
    image.show()  # Display the image
    # Or, save the image: image.save('image.png')

# Example usage: Extract the 10th image
input_file = '../inputs/t10k-images.idx3-ubyte'
output_file = '../inputs/image_0.ubyte'
image_index = 0

extract_mnist_image(input_file, output_file, image_index)
view_mnist_image(output_file)