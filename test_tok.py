from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-2B", trust_remote_code=True)
prompt = "<|im_start|>system\nYou are a rigorous logical reasoning engine.<|im_end|>\n"
tokens = tokenizer(prompt)["input_ids"]
print("Tokens:", tokens)
decoded = tokenizer.decode(tokens)
print("Decoded:", repr(decoded))

