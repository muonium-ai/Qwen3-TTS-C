#!/usr/bin/env python3
import json, struct, sys

def inspect_safetensors(path):
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    return tensors

# Main model
print("=== Main Model Tensors (selected) ===")
ts = inspect_safetensors('model/model.safetensors')
for name in sorted(ts.keys()):
    info = ts[name]
    if any(x in name for x in ['codec_embedding', 'text_embedding', 'text_projection', 
                                'codec_head', 'small_to_mtp', 'lm_head.0', 'lm_head.1',
                                'codec_embedding.0', 'codec_embedding.1',
                                'layers.0.self_attn.q', 'layers.0.self_attn.k', 
                                'layers.0.mlp.gate', 'model.norm']):
        print(f"  {name}: {info['dtype']} {info['shape']}")

print()
print("=== Speech Tokenizer Tensors (selected) ===")
ts2 = inspect_safetensors('model/speech_tokenizer/model.safetensors')
for name in sorted(ts2.keys()):
    info = ts2[name]
    if any(x in name for x in ['codebook', 'output_proj', 'pre_conv', 'input_proj',
                                'layers.0.self_attn.q', 'layers.0.mlp.gate',
                                'upsample.0', 'decoder.0', 'decoder.5', 'decoder.6',
                                'pre_transformer.norm']):
        print(f"  {name}: {info['dtype']} {info['shape']}")
