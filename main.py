from PIL import Image
from utils2 import Utils2

def process_inputs(prompt: str, image_path: str):
    # Display the prompt
    print("Your Prompt:")
    print(prompt)
    utils2_obj = Utils2()
    utils2_obj.llm_call(prompt)
    # print(res)
    # # Load and display the image
    # try:
    #     image = Image.open(image_path)
    #     image.show()  # This will open the image using the default image viewer
    # except Exception as e:
    #     print(f"Error loading image: {e}")



def main():
    prompt = "Describe the scene depicted in this image."
    image_path = "path/to/your/image.jpg" 
    
    process_inputs(prompt, image_path)

if __name__ == "__main__":
    main()
