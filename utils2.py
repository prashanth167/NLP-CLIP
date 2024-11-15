import json
import torch
from utils import Utils
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json

class Utils2:
    def __init__(self):
        self.utils_class = Utils(device="cuda")

    def encode_image(self,image_features):
        return self.utils_class.clip_encode_image(image_features)

    def encode_text(self,phrase):
        return self.utils_class.clip_encode_text(phrase)

    def subtract_embeddings(self, embedding1, embedding2):
        return embedding1 - embedding2  

    def add_embeddings(embedding1, embedding2):
        return embedding1 + embedding2

    def generate_image_from_embedding(self, embedding):
        return self.utils_class.text2image(embedding)

    def refine_image_with_phrase(self, image, phrase):
        return self.utils_class.ipadapter_text2image(phrase, image)


    def process_json(self,tasks):
        results = {}  
        
        for task in tasks:
            task_type = task.get("task")
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


    def llm_call(self,input_question):
        with open("prompt.txt", "r", encoding='utf-8') as file:
            input_question = file.read().strip() 
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set seed for reproducibility
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Load the tokenizer
        print("Loading tokenizer...")
        model_path = "/home/ptummal3/Downloads/clip/llama-3.1-transformers-8b-instruct-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

        # Add special tokens and resize embeddings
        special_tokens_dict = {'additional_special_tokens': ['<s>', '</s>', '[INST]', '[/INST]', '<<SYS>>', '<</SYS>>']}
        tokenizer.add_special_tokens(special_tokens_dict)

        # Define system message
        system_message = ""

        # Load the base model with 8-bit precision
        print("Loading model...")
        # bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # quantization_config=bnb_config,
            device_map=None  # Remove or set to None to load the model on a single GPU
        ).to(device)

        # Resize token embeddings in case new tokens were added
        model.resize_token_embeddings(len(tokenizer))

        # Set the padding token ID
        model.config.pad_token_id = tokenizer.pad_token_id

        # Function to format the input question
        def format_question(question):
            return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{question} [/INST]"

        # Prepare and format the single prompt
        prompt = format_question(input_question)

        # Tokenize prompt
        print("Tokenizing input...")
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=4096).to(device)

        # Generate output
        print("Generating output...")
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_answer = output_text[len(prompt):].strip()

        print(f"\nGenerated Answer: {generated_answer}")
        model.to('cpu')
        self.process_json(generated_answer)


    # Sample JSON input
    json_input = [
        {
            "task": "clip_model",
            "action": "encode_image",
            "input_variables": ["Image features from user input"],
            "output_variable": "Vθ"
        },
        {
            "task": "clip_model",
            "action": "encode_text",
            "input_phrases": ["dress"],
            "output_variable": "Tθ"
        },
        {
            "task": "compute",
            "action": "subtract",
            "input_variables": ["Vθ", "Tθ"],
            "output_variable": "Iθ"
        },
        {
            "task": "clip_model",
            "action": "encode_text",
            "input_phrases": ["a hat"],
            "output_variable": "T1"
        },
        {
            "task": "compute",
            "action": "add",
            "input_variables": ["Iθ", "T1"],
            "output_variable": "I2"
        },
        {
            "task": "text2image",
            "action": "generate_image",
            "input_variables": ["I2"],
            "output_variable": "GeneratedImage"
        },
        {
            "task": "ip_adapter",
            "action": "refine_image",
            "input_image": "GeneratedImage",
            "input_phrases": ["dog"],
            "output_variable": "FinalImage"
        }
    ]

    # output = process_json(json_input)
    # print("Final Output:", output)

            
