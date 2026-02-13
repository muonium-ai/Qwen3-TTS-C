#!/usr/bin/env python3
import json
with open('model/config.json') as f:
    cfg = json.load(f)
tc = cfg.get('talker_config', {})
cp = tc.get('code_predictor_config', {})
print('=== Talker Config ===')
for k in ['hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size','text_hidden_size','text_vocab_size','num_code_groups','rms_norm_eps','rope_theta']:
    print(f'  {k}: {tc.get(k)}')
print(f'  head_dim: {tc.get("hidden_size",0)//tc.get("num_attention_heads",1)}')
print(f'  mrope_section: {tc.get("rope_scaling",{}).get("mrope_section")}')
print()
print('=== Code Predictor Config ===')
for k in ['hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size','head_dim']:
    print(f'  {k}: {cp.get(k)}')
print()
print('=== Speaker Map ===')
spk = tc.get('spk_id', {})
for name, ids in spk.items():
    print(f'  {name}: {ids}')
print()
print('=== Language Map ===')
lang = tc.get('codec_language_id', {})
for name, lid in lang.items():
    print(f'  {name}: {lid}')
print()
with open('model/speech_tokenizer/config.json') as f:
    scfg = json.load(f)
dc = scfg.get('decoder_config', {})
print('=== Codec Decoder Config ===')
for k in ['hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','codebook_dim','codebook_size','latent_dim','num_quantizers','sliding_window','decoder_dim','rms_norm_eps','upsample_rates','upsampling_ratios']:
    print(f'  {k}: {dc.get(k)}')
