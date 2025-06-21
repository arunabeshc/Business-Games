import modal
from modal import App, Volume, Image

# Setup

app = modal.App("llama_3.1")
image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate")
secrets = [modal.Secret.from_name("hf-secret")]
GPU = "A100"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" 



@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def generate(prompt: str) -> str:
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

    # Quant Config
    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)


    # Load model and tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        quantization_config=quant_config,
        device_map="auto"
    )

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones(inputs.shape, device="cuda")
    outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5000, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
