import modal
from modal import App, Volume, Image

# Setup

app = modal.App("llama_3.1_assistant")
image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate")
secrets = [modal.Secret.from_name("hf-secret")]
GPU = "A100"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" 



@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def generate(prompt: str) -> str:
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)


    # Proper Llama-style chat template that expects assistant to respond once
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}{{ message['content'] }}\n\n{% endif %}"
        "{% if message['role'] == 'user' %}Human: {{ message['content'] }}\n\n{% endif %}"
        "{% if message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n\n{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}Assistant: {% endif %}"
    )

    inputs = tokenizer.apply_chat_template(
        prompt,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")

    # Create attention mask from non-pad tokens
    attention_mask = (inputs != tokenizer.pad_token_id).long().to("cuda")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", quantization_config=quant_config)

    # Create stop tokens for common chat patterns
    stop_strings = ["\nHuman:", "\nUser:", "\nuser:", "Human:", "User:"]
    stop_token_ids = []
    for stop_string in stop_strings:
        tokens = tokenizer.encode(stop_string, add_special_tokens=False)
        if tokens:
            stop_token_ids.extend(tokens)

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=5000,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # Add stop token ids to prevent continuing conversation
        bad_words_ids=[stop_token_ids] if stop_token_ids else None
    )

    # Extract only the generated response
    input_length = inputs.shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Clean up the response
    response = response.strip()
    # Remove any remaining conversation markers
    for stop_word in ["\nHuman:", "\nUser:", "\nuser:", "Human:", "User:"]:
        if stop_word in response:
            response = response.split(stop_word)[0]

    return(f"assistant:{response.strip()}")