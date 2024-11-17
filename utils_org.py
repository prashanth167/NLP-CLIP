import torch
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from diffusers import DiffusionPipeline

from ip_adapter import IPAdapterXL


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

        self.text2img: DiffusionPipeline = (
            DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
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
            prompt=text, ip_adapter_image=image, num_inference_steps=500,
            # negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
        ).images[0]
        self.text2img.to("cpu")
        self.text2img.unload_ip_adapter()

        return image