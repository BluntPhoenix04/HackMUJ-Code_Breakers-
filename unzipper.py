import zipfile
import os

def unzip_dataset(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print(f"Error: The file {zip_path} does not exist.")
        return
    
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Dataset extracted to: {extract_to}")

if __name__ == "__main__":
    unzip_dataset('dataset.zip', 'dataset')
