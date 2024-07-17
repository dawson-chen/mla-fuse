import torch
import os
import json
import re
from safetensors.torch import save_file
from safetensors import safe_open
from glob import glob

from modeling_deepseek import DeepseekV2ForCausalLM
from configuration_deepseek import DeepseekV2Config
import torch
from transformers import AutoConfig


model_path = f'./deepseek-ai/DeepSeek-V2-Lite/'
fused_model_path = './deepseek-ai/DeepSeek-V2-fuse1/'


os.makedirs(fused_model_path, exist_ok=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model_index = json.load(open(os.path.join(model_path, 'model.safetensors.index.json')))


def load_shard(file_name):
    static_dict = {}
    with safe_open(os.path.join(model_path, file_name), framework="pt", device='cpu') as f:
        for k in f.keys():
            static_dict[k] = f.get_tensor(k)
    return static_dict

def save_shard(shard_file, static_dict, model_index):
    for name in static_dict:
        model_index['metadata']['total_size'] += static_dict[name].numel() * 2
        model_index['weight_map'][name] = each
    save_file(
        static_dict, 
        shard_file, 
        {'format': 'pt'}
    )
    print(f'Save shard: {shard_file}')

def fuse_mla_weight(prefix, mla_weights, name_to_static_dict):
    # fuse $W_{UK}$ to $W_{UQ}$ or $W_Q$.
    if config.q_lora_rank:
        q_up_w = mla_weights[f'{prefix}q_b_proj.weight']
        q_in_dim = q_up_w.shape[-1]
    else:
        q_up_w = mla_weights[f'{prefix}q_proj.weight']
        q_in_dim = q_up_w.shape[-1]
    q_up_w = q_up_w.view(config.num_attention_heads, config.qk_nope_head_dim + config.qk_rope_head_dim, -1)
    q_up_w_nope = q_up_w[:, :config.qk_nope_head_dim]
    q_up_w_pe = q_up_w[:, config.qk_nope_head_dim:]

    kv_up_w_nope = mla_weights[f'{prefix}kv_b_proj.weight']
    kv_up_w_nope = kv_up_w_nope.view(config.num_attention_heads, config.qk_nope_head_dim + config.v_head_dim, -1)
    k_up_w_nope = kv_up_w_nope[:, :config.qk_nope_head_dim]
    v_up_w_nope = kv_up_w_nope[:, config.qk_nope_head_dim:]

    q_up_w_nope_fuse_k = torch.matmul(
        k_up_w_nope.transpose(1, 2).float(), 
        q_up_w_nope.float()
    ).to(q_up_w_nope.dtype)
    q_up_w_fuse_k = torch.concat([q_up_w_nope_fuse_k, q_up_w_pe], dim=1)
    q_up_w_fuse_k = q_up_w_fuse_k.reshape(-1, q_in_dim)

    # fuse $W_{UV}$ to $W_{O}$.
    o_w = mla_weights[f'{prefix}o_proj.weight']
    o_w_fused_v = torch.matmul(
        o_w.reshape(-1, config.num_attention_heads, config.v_head_dim).transpose(0, 1).contiguous().float(), 
        v_up_w_nope.float(),
    ).to(o_w.dtype)
    o_w_fused_v = o_w_fused_v.transpose(0, 1).contiguous().reshape(config.hidden_size, -1)

    del name_to_static_dict[f'{prefix}kv_b_proj.weight'][f'{prefix}kv_b_proj.weight']
    
    q_proj_key = f'{prefix}q_b_proj.weight' if config.q_lora_rank else f'{prefix}q_proj.weight'
    name_to_static_dict[q_proj_key][q_proj_key] = q_up_w_fuse_k
    name_to_static_dict[f'{prefix}o_proj.weight'][f'{prefix}o_proj.weight'] = o_w_fused_v


weight_map = model_index['weight_map']
model_index_new = {'metadata': {'total_size': 0}, 'weight_map': {}}
saved_shards = []
shard_to_max_layer = {}
for name, file in weight_map.items():
    if file not in shard_to_max_layer:
        shard_to_max_layer[file] = 0
    
    if 'layers.' in name:
        layer_idx = int(re.findall(f'layers\.(\d+)\.', name)[0])
        shard_to_max_layer[file] = max(shard_to_max_layer[file], layer_idx)
        

sharded_static_dict = {}

for i in range(config.num_hidden_layers):
    prefix = f'model.layers.{i}.self_attn.'
    
    shard_file_need = {weight_map[each] for each in weight_map if f'layers.{i}.self_attn' in each}
    
    mla_weights = {}
    name_to_static_dict = {}
    for each in shard_file_need:
        if each not in sharded_static_dict:
            static_dict = load_shard(each)
            sharded_static_dict[each] = static_dict
        static_dict = sharded_static_dict[each]
        
        for name in static_dict:
            name_to_static_dict[name] = static_dict
            if '.self_attn' in name:
                mla_weights[name] = static_dict[name]
    
    fuse_mla_weight(prefix, mla_weights, name_to_static_dict)
    print(f'do fuse_mla_weight to layer {i+1}')

    for each in sharded_static_dict:
        if shard_to_max_layer[each] < i and sharded_static_dict[each]:
            save_shard(os.path.join(fused_model_path, each), sharded_static_dict[each], model_index_new)
            saved_shards.append(each)
            # release shard ckpt
            sharded_static_dict[each] = {}

model_files = sorted(glob(os.path.join(model_path, 'model*.safetensors')))
model_files = [os.path.basename(each) for each in model_files]

for each in model_files:
    if each in saved_shards:
        continue
    if each in sharded_static_dict and sharded_static_dict[each]:
        static_dict = sharded_static_dict[each]
    else:
        static_dict = load_shard(os.path.join(model_path, each))
    save_shard(os.path.join(fused_model_path, each), static_dict, model_index_new)

json.dump(model_index_new, open(os.path.join(fused_model_path, 'model.safetensors.index.json'), 'w'))

config.fuse_weight = True
config.to_json_file(os.path.join(fused_model_path, 'config.json'))
print('set \'config.fuse_weight = True\' and save')
