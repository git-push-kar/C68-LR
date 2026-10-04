import torch
from polyvalent_lr.config import LRModuleConfig, LoRAConfig
from polyvalent_lr.models.lora_setup import setup_internvl_lr_lora
from transformers import AutoTokenizer
import time

print('Testing single forward/backward pass on RTX A5000...')
config = LoRAConfig(r=32, lora_alpha=64)
model, tokenizer = setup_internvl_lr_lora(
    'OpenGVLab/InternVL3-2B', config, device_map=None, torch_dtype=torch.bfloat16
)
model = model.to('cuda:0')
model.train()

bsz = 32
seq_len = 384
input_ids = torch.randint(0, 1000, (bsz, seq_len)).cuda()
labels = torch.randint(0, 1000, (bsz, seq_len)).cuda()

print(f'Starting warmup pass...')
# Warmup
out = model(input_ids=input_ids, labels=labels)
loss = out.loss
loss.backward()
print(f'Warmup done.')

torch.cuda.synchronize()
start = time.time()
out = model(input_ids=input_ids, labels=labels)
loss = out.loss
loss.backward()
torch.cuda.synchronize()
end = time.time()

print(f'Time for 1 pass (bsz=32, seq_len=384): {end - start:.4f} seconds')
print(f'Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB')
print(f'Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB')
