import torch
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from diffusers import DiffusionPipeline
from ip_adapter import IPAdapterXL
import json
import re
from peft import PeftModel


class Utils:
    def __init__(self, device="cuda"):

        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text_encoder = CLIPTextModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.image_processor = AutoImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.text2img: DiffusionPipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        self.device = device

        # self.text2img.to(self.device)

    def clip_encode_text(self, text):
        text = self.tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True
        )
        text = {k: v.to(self.device) for k, v in text.items()}

        self.clip_text_encoder.to(self.device)
        text_features = self.clip_text_encoder(**text).text_embeds
        self.clip_text_encoder.to("cpu")

        return text_features

    def clip_encode_image(self, image):
        image = self.image_processor(images=image, return_tensors="pt")
        image = {k: v.to(self.device) for k, v in image.items()}

        self.clip_image_encoder.to(self.device)
        image_features = self.clip_image_encoder(**image).image_embeds
        self.clip_image_encoder.to("cpu")

        return image_features

    def text2image(self, text):
        self.text2img.to(self.device)
        # if self.text2img is None:
        #     print("Class not working")
        image = self.text2img(text, num_inference_steps=50).images[0]
        self.text2img.to("cpu")

        return image

    def ipadapter_text2image(self, text, image=None):
        self.text2img.load_ip_adapter(
            "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
        )
        self.text2img.to(self.device)

        image = self.text2img(
            prompt=text,
            ip_adapter_image=image,
            num_inference_steps=500,
            # negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
        ).images[0]
        self.text2img.to("cpu")
        self.text2img.unload_ip_adapter()

        return image

    def encode_image(self, image_features):
        return self.clip_encode_image(image_features)

    def encode_text(self, phrase):
        return self.clip_encode_text(phrase)

    def subtract_embeddings(self, embedding1, embedding2):
        return embedding1 - embedding2

    def add_embeddings(embedding1, embedding2):
        return embedding1 + embedding2

    def generate_image_from_embedding(self, embedding):
        return self.text2image(embedding)

    def refine_image_with_phrase(self, image, phrase):
        return self.ipadapter_text2image(phrase, image)

    def process_json(self, tasks):
        print(tasks)
        results = {}
    
        for task in tasks:
            print(task)
            task_type = task.get("function")
            action = task.get("action")

            if task_type == "clip_model":
                if action == "encode_image":
                    input_data = task["input_variables"][0]
                    output = self.encode_image(input_data)
                    results[task["output_variable"]] = output

                elif action == "encode_text":
                    input_phrase = task["input_phrases"][0]
                    output = self.encode_text(input_phrase)
                    results[task["output_variable"]] = output

            elif task_type == "compute":
                var1, var2 = task["input_variables"]
                if action == "subtract":
                    output = self.subtract_embeddings(results[var1], results[var2])
                    results[task["output_variable"]] = output

                elif action == "add":
                    output = self.add_embeddings(results[var1], results[var2])
                    results[task["output_variable"]] = output

            elif task_type == "text2image" and action == "generate_image":
                input_var = task["input_variables"][0]
                output = self.generate_image_from_embedding(results[input_var])
                results[task["output_variable"]] = output

            elif task_type == "ip_adapter" and action == "refine_image":
                input_image = results[task["input_image"]]
                input_phrase = task["input_phrases"][0]
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
        with open("prompt2.txt", "r", encoding="utf-8") as file:
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
            "/home/ptummal3/Downloads/clip/llama-3.1-transformers-8b-instruct-v2"
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
