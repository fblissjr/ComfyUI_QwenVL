import os
import torch
# keeping only necessary base imports from transformers for Qwen2 node
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import numpy as np
import folder_paths
import uuid

# Imports needed for the modified node
import requests  # For making HTTP requests
import base64  # For encoding images
import io  # For handling image bytes
import json  # For parsing stop sequences


# tensor_to_pil function remains the same
def tensor_to_pil(image_tensor, batch_index=0) -> Image.Image:
    """Converts a ComfyUI IMAGE tensor to a PIL Image."""
    image_tensor_cpu = image_tensor[batch_index].cpu()
    i = 255.0 * image_tensor_cpu.numpy()
    img_array = np.clip(i, 0, 255).astype(np.uint8)
    if img_array.ndim == 4 and img_array.shape[0] == 1:
        img_array = img_array.squeeze(0)
    if img_array.ndim == 2:
        img = Image.fromarray(img_array, "L").convert("RGB")
    elif img_array.ndim == 3:
        img = Image.fromarray(img_array)
    else:
        raise ValueError(
            f"Unexpected image tensor shape after processing: {img_array.shape}"
        )
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# Qwen2VL_Remote class remains the same as the previous version (it never downloaded)
class Qwen2VL_Remote:
    def __init__(self):
        self.server_url = None

    @classmethod
    def INPUT_TYPES(cls):
        # Input types remain the same
        return {
            "required": {
                "server_url": ("STRING", {"default": "http://macos_server_ip:8080"}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 4096, "step": 1},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "stop_sequences": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL/Remote"

    def inference(
        self,
        server_url,
        text,
        image,
        temperature,
        top_p,
        max_new_tokens,
        repetition_penalty,
        system_prompt=None,
        stop_sequences=None,
    ):
        # Inference logic remains the same - calling the remote API
        if not server_url or not server_url.startswith(("http://", "https://")):
            return ("Error: Invalid server_url provided.",)
        self.server_url = server_url.strip("/")
        api_endpoint = f"{self.server_url}/v1/chat/completions"
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        user_content = [{"type": "text", "text": text}]
        num_images = image.shape[0] if image is not None else 0
        if num_images == 0:
            return ("Error: Image input is required for Qwen2VL_Remote node.",)
        print(f"[Qwen2VL_Remote] Processing {num_images} image(s).")
        for i in range(num_images):
            try:
                pil_image = tensor_to_pil(image, batch_index=i)
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG", quality=90)
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                base64_image_data = f"data:image/jpeg;base64,{base64_image}"
                user_content.append(
                    {"type": "image_url", "image_url": {"url": base64_image_data}}
                )
            except Exception as e:
                return (f"Error processing image {i + 1}/{num_images}: {str(e)}",)
        messages.append({"role": "user", "content": user_content})
        payload = {
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if top_p < 1.0:
            payload["top_p"] = top_p
        if repetition_penalty != 1.0:
            payload["repetition_penalty"] = repetition_penalty
        if stop_sequences and stop_sequences.strip():
            stops = [s.strip() for s in stop_sequences.splitlines() if s.strip()]
            if stops:
                payload["stop"] = stops
        try:
            print(f"[Qwen2VL_Remote] Sending request to: {api_endpoint}")
            debug_payload = json.loads(json.dumps(payload))
            for msg in debug_payload.get("messages", []):
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image_url":
                            item["image_url"]["url"] = f"data:image/jpeg;base64,<...>"
            print(f"[Qwen2VL_Remote] Payload (images truncated): {debug_payload}")
            response = requests.post(
                api_endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            print(
                f"[Qwen2VL_Remote] Server Response Status Code: {response.status_code}"
            )
            response.raise_for_status()
            response_data = response.json()
            print(f"[Qwen2VL_Remote] Received response data: {response_data}")
            if "choices" in response_data and len(response_data["choices"]) > 0:
                message = response_data["choices"][0].get("message", {})
                result_text = message.get(
                    "content", "Error: Could not extract content."
                )
                return (result_text,)
            elif "error" in response_data:
                error_msg = response_data["error"].get(
                    "message", "Unknown server error"
                )
                print(f"[Qwen2VL_Remote] Server returned error: {error_msg}")
                return (f"Error from server: {error_msg}",)
            else:
                print(f"[Qwen2VL_Remote] Unexpected response format: {response_data}")
                return ("Error: Unexpected response format.",)
        except requests.exceptions.Timeout:
            print(f"[Qwen2VL_Remote] Request timed out.")
            return (f"Error: Request timed out connecting to {self.server_url}",)
        except requests.exceptions.RequestException as e:
            print(f"[Qwen2VL_Remote] Connection Error: {str(e)}")
            return (f"Error connecting to {self.server_url}: {str(e)}",)
        except Exception as e:
            print(f"[Qwen2VL_Remote] Unexpected Error: {str(e)}")
            import traceback

            traceback.print_exc()
            return (f"An unexpected error occurred: {str(e)}",)


# --- Modified Qwen2 text-only node (NO DOWNLOAD) ---
class Qwen2:
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        # Keep original inputs
        return {
            "required": {
                "system": (
                    "STRING",
                    {"default": "You are a helpful assistant.", "multiline": True},
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen2.5-3B-Instruct",
                        "Qwen2.5-7B-Instruct",
                        "Qwen2.5-14B-Instruct",
                        "Qwen2.5-32B-Instruct",
                    ],
                    {"default": "Qwen2.5-7B-Instruct"},
                ),
                "quantization": (["none", "4bit", "8bit"], {"default": "none"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"  # Keep original category

    def inference(
        self,
        system,
        prompt,
        model,
        quantization,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        seed,
    ):
        if not prompt.strip():
            return ("Error: Prompt input is empty.",)
        if seed != -1:
            torch.manual_seed(seed)

        # Construct the expected model path *without* downloading
        model_id = f"qwen/{model}"
        expected_model_path = os.path.join(
            folder_paths.models_dir, "LLM", os.path.basename(model_id)
        )

        # --- Check if model exists locally ---
        if not os.path.exists(expected_model_path):
            return (
                f"Error: Model not found locally at {expected_model_path}. Please download it manually.",
            )
        # --- End check ---

        # Assign the path if it exists
        self.model_checkpoint = expected_model_path

        # Load tokenizer and model only if needed
        try:
            if self.tokenizer is None:
                print(f"[Qwen2 Node] Loading tokenizer from: {self.model_checkpoint}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

            if self.model is None:
                print(
                    f"[Qwen2 Node] Loading model from: {self.model_checkpoint} with quantization: {quantization}"
                )
                q_config = None
                if quantization == "4bit":
                    q_config = BitsAndBytesConfig(load_in_4bit=True)
                elif quantization == "8bit":
                    q_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_checkpoint,
                    torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                    device_map="auto",
                    quantization_config=q_config,
                )
                print(
                    f"[Qwen2 Node] Model loaded successfully on device: {self.model.device}"
                )

        except Exception as e:
            self.tokenizer = None  # Reset on error
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print(f"[Qwen2 Node] Error loading model/tokenizer: {str(e)}")
            import traceback

            traceback.print_exc()
            return (
                f"Error loading model/tokenizer from {self.model_checkpoint}: {str(e)}",
            )

        # Inference logic remains the same
        with torch.no_grad():
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(
                self.model.device
            )
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=max_new_tokens, temperature=temperature
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids)
            ]
            result = self.tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            if not keep_model_loaded:
                print(f"[Qwen2 Node] Unloading model and tokenizer.")
                del self.tokenizer, self.model
                self.tokenizer, self.model = None, None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            return (result,)


NODE_CLASS_MAPPINGS = {"Qwen2.5VL_Remote": Qwen2VL_Remote, "Qwen2.5": Qwen2}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2.5VL_Remote": "Qwen2.5VL (Remote OpenAI API)",
    "Qwen2.5": "Qwen2.5 (Local Text)",
}