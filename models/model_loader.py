import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from models.attention_scaling import AttentionScaling

class ModelLoader:
    def __init__(self, model_name, ntk_scale, pi_window_size, device='cpu'):
        self.model = LlamaForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.scaler = AttentionScaling(ntk_scale, pi_window_size)
        self.device = device

    def apply_attention_scaling(self, input_ids):
        embeddings = self.model.get_input_embeddings()(input_ids).to(self.device)
        embeddings = self.scaler.ntk_rope(embeddings)
        return self.scaler.position_interpolation(embeddings)

    def forward(self, input_ids):
        embeddings = self.apply_attention_scaling(input_ids)
        outputs = self.model(inputs_embeds=embeddings)
        return outputs.logits
    