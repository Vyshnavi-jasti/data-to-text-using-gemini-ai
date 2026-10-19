
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading Qwen model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful data analyst. "
            "Answer clearly and use only the information provided."
        )
    },
    {
        "role": "user",
        "content": "Give me a short introduction to large language models."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer(
    [text],
    return_tensors="pt"
).to(model.device)

with torch.inference_mode():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=150,
        do_sample=False
    )

new_tokens = generated_ids[:, model_inputs.input_ids.shape[1]:]

response = tokenizer.batch_decode(
    new_tokens,
    skip_special_tokens=True
)[0]

print("\nQwen response:\n")
print(response)
