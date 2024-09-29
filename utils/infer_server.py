import requests
import json
import os
import numpy as np

def load_mnist_image(filename):
    """Loads a single MNIST image from a .ubyte file."""
    with open(filename, 'rb') as f:
        # Read image data as bytes
        image_data = f.read(28 * 28) 
        # Convert to numpy array, reshape, and normalize
        image = np.frombuffer(image_data, dtype=np.uint8).reshape(1, 1, 28, 28).astype(np.float32)
    return image


def process_image(filename, url):
    image = load_mnist_image(filename)
    flattened_data = image.flatten().tolist()
    
    data = {"data": flattened_data}

    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()["result"]
        print(f"Image is the number {np.argmax(result)}:", result)
    else:
        print(f"Error for {filename}:", response.status_code, response.text)
        print("Response content:", response.content)

if __name__ == "__main__":
    inputs_folder = "/home/michal/code/inference_engine/inputs"
    url = "http://localhost:8080/infer"

    # List all files in the inputs folder
    files = os.listdir(inputs_folder)
    
    for file in sorted(files):
        full_path = os.path.join(inputs_folder, file)
        print(f"Processing file: {file}")
        process_image(full_path, url)
        print("------------------------")

    print("All images processed.")