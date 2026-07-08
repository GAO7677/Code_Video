from torch import nn
import torch
from transformers import AutoTokenizer, UMT5EncoderModel
import os
import html
from typing import Any, Callable, Dict, List, Optional, Union
import regex as re
import ftfy

# tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
# model = UMT5EncoderModel.from_pretrained("google/umt5-small")
# input_ids = tokenizer(
#     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
# ).input_ids  # Batch size 1
# outputs = model(input_ids=input_ids)
# last_hidden_states = outputs.last_hidden_state


def basic_clean(text):
  text = ftfy.fix_text(text)
  text = html.unescape(html.unescape(text))
  return text.strip()


def whitespace_clean(text):
  text = re.sub(r"\s+", " ", text)
  text = text.strip()
  return text


def prompt_clean(text):
  text = whitespace_clean(basic_clean(text))
  return text

class WanTextEncoderWrapper(nn.Module):

  def __init__(self, 
      pretrained_model_name_or_path, 
      device,
      dtype):
      super().__init__()   

      tokenizer_path = os.path.join(pretrained_model_name_or_path,"tokenizer")
      encoder_path = os.path.join(pretrained_model_name_or_path,"text_encoder")
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
      self.text_encoder = UMT5EncoderModel.from_pretrained(encoder_path).to(device).to(dtype)
    
  def _get_t5_prompt_embeds(
      self,
      prompt: Union[str, List[str]] = None,
      num_videos_per_prompt: int = 1,
      max_sequence_length: int = 226,
      device: Optional[torch.device] = None,
      dtype: Optional[torch.dtype] = None,
  ):
      device = device
      dtype = dtype or self.text_encoder.dtype

      prompt = [prompt] if isinstance(prompt, str) else prompt
      prompt = [prompt_clean(u) for u in prompt]
      batch_size = len(prompt)

      text_inputs = self.tokenizer(
          prompt,
          padding="max_length",
          max_length=max_sequence_length,
          truncation=True,
          add_special_tokens=True,
          return_attention_mask=True,
          return_tensors="pt",
      )
      text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
      seq_lens = mask.gt(0).sum(dim=1).long()

      prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
      prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
      prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
      prompt_embeds = torch.stack(
          [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
      )

      # duplicate text embeddings for each generation per prompt, using mps friendly method
      _, seq_len, _ = prompt_embeds.shape
      prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
      prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

      return prompt_embeds,mask

  @torch.no_grad()
  def __call__(self,
      prompt : str,
      max_sequence_length : int = 512,
      device : torch.device = torch.device("cpu"),
      dtype : torch.dtype = torch.bfloat16,
      ):
      prompt = [prompt] if isinstance(prompt, str) else prompt

      prompt_embeds,mask = self._get_t5_prompt_embeds(
          prompt=prompt,
          num_videos_per_prompt=1,
          max_sequence_length=max_sequence_length,
          device=device,  
          dtype=dtype,
      )
      return prompt_embeds,mask

if __name__ == "__main__":
  device = torch.device("cuda")
  text_encoder = WanTextEncoderWrapper("/video_hy2/workspace/zhangqiyuan.zqy/pt/Wan2.1-T2V-1.3B-diffusers",device,torch.bfloat16)
  prompt_embeds,mask=text_encoder("one clown tries to male yoda eat falawel at ju country",512,device)
  print(mask.shape)
  print(prompt_embeds.shape)