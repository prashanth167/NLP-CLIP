import torch
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from ip_adapter import IPAdapterXL
import json
import re
from peft import PeftModel
from PIL import Image

from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
import torch

from diffusers import AutoPipelineForText2Image



class Utils:
    def __init__(self, device="cuda:0"):

        # self.tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        # self.clip_text_encoder = CLIPTextModelWithProjection.from_pretrained(
        #     "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        # )

        # self.image_processor = AutoImageProcessor.from_pretrained(
        #     "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        # )
        # self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        #     "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        # )

        # self.text2img: DiffusionPipeline = StableDiffusionPipeline.from_pretrained(
        #     # "stabilityai/stable-diffusion-xl-base-1.0",
        #     "stabilityai/stable-diffusion-2-1",
        #     torch_dtype=torch.float16,
        #     use_safetensors=True,
        #     variant="fp16",
        # )
        
        # Kandinsky model
        self.prior_pipeline = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16)
        self.pipeline = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16)

        self.clip_vision_model = self.prior_pipeline.image_encoder
        self.preprocess = self.prior_pipeline.image_processor

        
        # IP Adapter
        self.sd_pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        self.sd_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        self.sd_pipeline.set_ip_adapter_scale(0.6)

        self.device = device

        self.results = { 'V0': '', 'T0': '', 'T1': '', 'I0': '', 'I2': '' }
        
    def clip_encode_text(self, text):
        # text = self.tokenizer(
        #     text, return_tensors="pt", padding="max_length", truncation=True
        # )
        # text = {k: v.to(self.device) for k, v in text.items()}

        # self.clip_text_encoder.to(self.device)
        # outputs = self.clip_text_encoder(**text)
        # text_features = outputs.text_embeds
        # last_hidden_state = outputs.last_hidden_state
        # self.clip_text_encoder.to("cpu")

        # return text_features, last_hidden_state
        
        negative_prompt = "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"
        self.prior_pipeline.to(self.device)
        image_embeds2, _ = self.prior_pipeline(text, negative_prompt, guidance_scale=1.0, num_inference_steps=100, generator=torch.Generator().manual_seed(42)).to_tuple()
        # self.prior_pipeline.to("cpu")
        return image_embeds2

    def clip_encode_image(self, image):
        # image = self.image_processor(images=image, return_tensors="pt")
        # image = {k: v.to(self.device) for k, v in image.items()}

        # self.clip_image_encoder.to(self.device)
        # image_features = self.clip_image_encoder(**image).image_embeds
        # self.clip_image_encoder.to("cpu")

        # return image_features
        # image = self.preprocess(images=image, return_tensors="pt").to("cuda")
        # self.clip_vision_model.to("cuda")
        # print("aaa", self.clip_vision_model(image))
        # image_features = self.clip_vision_model(image).image_embeds
        # self.clip_vision_model.to("cpu")
        # return image_features

        inputs = self.preprocess(images=image, return_tensors="pt").to(self.device)
        print("Processed image tensor shape:", inputs['pixel_values'].shape)
        self.clip_vision_model.to("cuda:0")
        with torch.no_grad():
            outputs = self.clip_vision_model(pixel_values=inputs['pixel_values'])
            image_features = outputs.image_embeds
        # self.clip_vision_model.to("cpu")
        return image_features

    def text2image(self, image_embeddings):
        # negative_text_embeddings, negative_last_hidden_states = self.clip_encode_text("")
        # self.text2img.to(self.device)
        # # text_embeddings = text_embeddings
        # image = self.text2img(prompt_embeds=text_embeddings, negative_prompt_embeds=negative_last_hidden_states, num_inference_steps=50).images[0]
        # self.text2img.to("cpu")

        # return image
        _, negative_image_embeds =  self.prior_pipeline("", "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality", guidance_scale=2.0, num_inference_steps=20, generator=torch.Generator().manual_seed(42)).to_tuple()
        self.pipeline.to(self.device)
        image = self.pipeline(image_embeds=image_embeddings, negative_image_embeds=negative_image_embeds, height=768, width=768, generator=torch.Generator().manual_seed(42)).images[0]
        # self.pipeline.to("cpu")
        return image 

    def ipadapter_text2image(self, text, image=None):
        # self.text2img.load_ip_adapter(
        #     "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
        # )
        # self.text2img.to(self.device)

        # image = self.text2img(
        #     prompt=text,
        #     ip_adapter_image=image,
        #     num_inference_steps=500,
        #     # negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
        # ).images[0]
        # self.text2img.to("cpu")
        # self.text2img.unload_ip_adapter()

        # return image
        self.sd_pipeline.to(self.device)
        images = self.sd_pipeline(
            prompt=text,
            ip_adapter_image=image,
            negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
            num_inference_steps=50,
            generator=torch.Generator().manual_seed(42),    
        ).images
        try:
            image = images[0]
            if isinstance(image, Image.Image):  # Check if the returned object is indeed a PIL Image
                image.save("final_image.jpg")
                print("Image saved successfully as 'final_image.jpg'.")
            else:
                print("Error: No valid image was returned from text2image.")
        except Exception as e:
            print(f"An error occurred while saving the image: {e}")
        return images[0]

    def encode_image(self, image_features):
        return self.clip_encode_image(image_features)

    def encode_text(self, phrase):
        return self.clip_encode_text(phrase)

    def subtract_embeddings(self, embedding1, embedding2):
        return embedding1 - embedding2

    def add_embeddings(self, embedding1, embedding2):
        return embedding1 + embedding2

    def generate_image_from_embedding(self, embedding):
        try:
            image = self.text2image(embedding)
            if isinstance(image, Image.Image):  # Check if the returned object is indeed a PIL Image
                image.save("output_image1.jpg")
                print("Image saved successfully as 'output_image1.jpg'.")
            else:
                print("Error: No valid image was returned from text2image.")
            return image
        except Exception as e:
            print(f"An error occurred while saving the image: {e}")
        


    def refine_image_with_phrase(self, image, phrase):
        return self.ipadapter_text2image(phrase, image)

    def process_json(self, tasks):
        
        # image encoding
        image = Image.open('red_dress.jpg')
        self.results['V0'] = self.clip_encode_image(image)
        print("image encodings", self.results['V0'])
        # print(self.results.keys())

        print("tasks", tasks)


        for task_name, task in tasks.items():
            task_type = task.get("function")
            action = task.get("action")
            input_text = task.get("input_text")
            input_variables = task.get("input_variables")
            output_variable = task.get("output_variable")

            print(task_type, action, input_text, input_variables, output_variable)
            


            if task_type == "clip_encode_text":
                output = self.encode_text(input_text[0])
                print("clip_encode_text", output)
                self.results[output_variable] = output


            elif task_type == "compute":
                var1, var2 = input_variables[0], input_variables[1]
                if action == "subtract":
                    output = self.subtract_embeddings(self.results[var1], self.results[var2])
                    print("subtraction", output)
                    self.results[output_variable] = output

                elif action == "add":
                    output = self.add_embeddings(self.results[var1], self.results[var2])
                    self.results[output_variable] = output
                    print("addition", output)
                    print(self.results)

            elif task_type == "text2image":
                input_var = input_variables[0]
                output = self.generate_image_from_embedding(self.results[input_var])
                print("text_2_image", output)
                self.results[output_variable] = output

            elif task_type == "ipadapter_text2image":
                input_image = self.results['Generated_Image']
                input_phrase = input_text[0]
                output = self.refine_image_with_phrase(input_image, input_phrase)

    def remove_extra_spaces(self, generated_answer):
        cleaned = re.sub(r'(?<!\w) +| +(?!\w)', '', generated_answer)
        cleaned = re.sub(r'[\t\n\r\f\v]+', '',cleaned)
        return cleaned

    def extract_json(self, json_text):
        # json_pattern = r"\[\{\"tasks\":.*[\]\}]"
        json_pattern = r"\[\{.*"
        incomplete_json = re.findall(json_pattern, json_text)
        if not incomplete_json:
            raise ValueError("No JSON found matching the pattern.")
        return incomplete_json[-1]

    def balance_json(self, incomplete_json):
        stack = []
        balanced_json = incomplete_json

        for char in incomplete_json:
            if char in "[{":
                stack.append(char)
            elif char in "]}":
                if stack and (
                    (char == "]" and stack[-1] == "[")
                    or (char == "}" and stack[-1] == "{")
                ):
                    stack.pop()
                else:
                    stack.append(char)

        for unmatched in reversed(stack):
            if unmatched == "[":
                balanced_json += "]"
            elif unmatched == "{":
                balanced_json += "}"

        return balanced_json

    def parse_json(self, balanced_json):
        return json.loads(balanced_json)[0]

    def clean_output(self, generated_answer):
        json_text = self.remove_extra_spaces(generated_answer)
        incomplete_json = self.extract_json(json_text)
        balanced_json = self.balance_json(incomplete_json)
        final_json = self.parse_json(balanced_json)
        return final_json

    def format_question(self, question, system_message):
        return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{question} [/INST]"

    def llm_call(self, input_question):
        with open("prompt3.txt", "r", encoding="utf-8") as file:
            input_question = file.read().strip()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set seed for reproducibility
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Load the tokenizer
        print("Loading tokenizer...")
        model_path = (
            "/home/vkanakav/Downloads/clip/llama-3.1-transformers-8b-instruct-v2"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Add special tokens and resize embeddings
        special_tokens_dict = {
            "additional_special_tokens": [
                "<s>",
                "</s>",
                "[INST]",
                "[/INST]",
                "<<SYS>>",
                "<</SYS>>",
            ]
        }
        tokenizer.add_special_tokens(special_tokens_dict)

        # Define system message
        system_message = ""

        # Load the base model with 8-bit precision
        print("Loading model...")
        # bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # quantization_config=bnb_config,
            device_map=None,  # Remove or set to None to load the model on a single GPU
        ).to(device)

        # Resize token embeddings in case new tokens were added
        model.resize_token_embeddings(len(tokenizer))

        # Set the padding token ID
        model.config.pad_token_id = tokenizer.pad_token_id

        # Function to format the input question

        # Prepare and format the single prompt
        prompt = self.format_question(input_question, system_message)

        # Tokenize prompt
        print("Tokenizing input...")
        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096
        ).to(device)

        # Generate output
        print("Generating output...")
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # generated_answer = output_text[len(prompt) :].strip()

        # print("Generated Text : ",  output_text)

        cleaned_output = self.clean_output(output_text)
        # print("clened_output: ", cleaned_output)
        model.to("cpu")
        self.process_json(cleaned_output)

if __name__ == "__main__":
    utils = Utils("cuda:0")
    prompt = "A dog wearing a hat in the style of the dress."
    image_path = "/home/cr8dl-user/abhiram/Zero-shot-Appearance-Transfer-using-CLIP-Abstraction/red_dress.jpg" 
    
    from PIL import Image
    img = Image.open(image_path)
    utils.text2img.to("cuda:0")

    text_embeds, last_hidden_state = utils.clip_encode_text(prompt)
    img = utils.text2image(last_hidden_state)
    img.save("output.jpg")
    # img = utils.text2img(prompt)
    # utils.text2img.to("cuda:1")
    # prompt_embeds, _ = utils.text2img.encode_prompt(prompt, device="cuda:1", num_images_per_prompt=1, do_classifier_free_guidance=True)
    # print(prompt_embeds.shape)