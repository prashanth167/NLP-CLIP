from PIL import Image
from utils import Utils
import os

def process_inputs(final_image_desc: str, folder_path: str):

    utils_obj = Utils()

    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        print("Image Path:")
        print(image_path)
        utils_obj.llm_call(final_image_desc, image_path)


if __name__ == "__main__":
    final_image_desc = "Final target image description : 'A dog wearing a hat that has a style of tshirt'"
    folder_path = "cloth"
    
    process_inputs(final_image_desc, folder_path)