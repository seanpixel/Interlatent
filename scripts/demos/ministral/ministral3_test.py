import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

model_id = "mistralai/Ministral-3-14B-Instruct-2512"

tokenizer = MistralCommonBackend.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Mistral3ForConditionalGeneration.from_pretrained(model_id, device_map={"": device})

image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What should i do when I am super caffeinated?",
            }
        ],
    },
]

tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

# Ensure every tensor is on the same device as the model (position_ids/attention_mask included)
for k, v in tokenized.items():
    if torch.is_tensor(v):
        if k == "pixel_values":
            tokenized[k] = v.to(dtype=torch.bfloat16, device=device)
        else:
            tokenized[k] = v.to(device=device)

output = model.generate(
    **tokenized,
    max_new_tokens=512,
)[0]

decoded_output = tokenizer.decode(output[len(tokenized["input_ids"][0]):])
print(decoded_output)
