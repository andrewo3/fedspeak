import torch
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "./llama/llama-2-7b-chat-hf"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device,torch.version.cuda)


access_token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_dir,token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_dir,token=access_token,device_map="auto",torch_dtype = torch.bfloat16)

prompt = "As a data scientist, can you explain the concept of regularization in machine learning?"
messages = [{
    "role":"user",
    "content":prompt
}]

model_inputs = tokenizer.apply_chat_template(messages,return_tensors = "pt").to(device)

out = model.generate(model_inputs,max_new_tokens = 1000, do_sample=True,pad_token_id=tokenizer.eos_token_id)

decoded = tokenizer.batch_decode(out)

print(decoded)