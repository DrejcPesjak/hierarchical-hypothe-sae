# %% import libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from functools import partial
import torch.nn as nn

# %% load model and tokenizer
torch.set_grad_enabled(False) # avoid blowing up mem

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_type=torch.float16,
)

# model = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-3-4b-it",
#     device_map='auto',
# )
# tokenizer =  AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    device_map='auto',
    quantization_config=bnb_config,
)
tokenizer =  AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
tokenizer.pad_token = tokenizer.eos_token

# %% test input
def format_prompt(user_prompt: str) -> str:
  return f"""<start_of_turn>user
{user_prompt}<end_of_turn>
<start_of_turn>model
"""

user_prompt = "Who is the best basketball player of all time?"
inputs = tokenizer.encode(format_prompt(user_prompt), return_tensors="pt", add_special_tokens=True).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))


# %% Gemma-Scope-2
LAYER = 22  # options are {9, 17, 22, 29}
WIDTH = "16k"   # options are {16k, 65k, 262k}
L0 = "medium"  # options are {small, medium, big}

path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2-4b-it",
    filename=f"resid_post/layer_{LAYER}_width_{WIDTH}_l0_{L0}/params.safetensors",
)

params = load_file(path_to_params)

# %% define sae
class JumpReLUSAE(nn.Module):
  def __init__(self, d_in, d_sae, affine_skip_connection=False):
    # Note that we initialise these to zeros because we're loading in pre-trained weights.
    # If you want to train your own SAEs then we recommend using blah
    super().__init__()
    self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
    self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
    self.threshold = nn.Parameter(torch.zeros(d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.b_dec = nn.Parameter(torch.zeros(d_in))
    if affine_skip_connection:
      self.affine_skip_connection = nn.Parameter(torch.zeros(d_in, d_in))
    else:
      self.affine_skip_connection = None

  def encode(self, input_acts):
    pre_acts = input_acts @ self.w_enc + self.b_enc
    mask = (pre_acts > self.threshold)
    acts = mask * torch.nn.functional.relu(pre_acts)
    return acts

  def decode(self, acts):
    return acts @ self.w_dec + self.b_dec

  def forward(self, x):
    acts = self.encode(x)
    recon = self.decode(acts)
    if self.affine_skip_connection is not None:
      return recon + x @ self.affine_skip_connection
    return recon

# %% load sae
d_model, d_sae = params["w_enc"].shape
sae = JumpReLUSAE(d_model, d_sae)
sae.load_state_dict(params)
sae.cuda()

# %% gather activations
def gather_acts_hook(mod, inputs, outputs, cache: dict, key: str, use_input: bool):
  """Generic hook function whic stores activations (either input or output of a particular PyTorch module)."""
  acts = inputs[0].squeeze(0) if use_input else outputs[0]  # inputs usually have a batch dim
  cache[key] = acts
  return outputs


def gather_residual_activations(model, target_layer, inputs):

  cache = {}

  # Add a hook function to store the output of this layer of the model
  handle = model.model.language_model.layers[target_layer].register_forward_hook(
      partial(gather_acts_hook, cache=cache, key="resid_post", use_input=False)
  )

  # Forward pass inside a try/except/finally block (useful just in case our hook breaks
  # and we can't remove it!)
  try:
    _ = model.forward(inputs)
  finally:
    handle.remove()

  return cache["resid_post"]

# %% get activations and run through sae
target_act = gather_residual_activations(model, LAYER, inputs.to(model.device))

sae_acts = sae.encode(target_act.to(torch.float32))
recon = sae.decode(sae_acts)
# %% get top activations
top_acts, top_latents = sae_acts.squeeze().mean(0).topk(5)

for act, idx in zip(top_acts, top_latents):
  print(f"{act:>6.1f} | {idx}")

# %% feature logits
feature_idx = 118

w_u = model.lm_head.weight  # shape (d_vocab, d_model)
w_u_eff = w_u * model.model.language_model.norm.weight

decoder_vector = sae.w_dec[feature_idx].to(w_u_eff.dtype) # shape (d_model,)

top_activations, top_tokens = torch.topk(w_u_eff @ decoder_vector, k=10)

for act, tok in zip(top_activations, top_tokens):
  print(f"{act:.4f} | {tokenizer.decode(tok)}")


# %% feature neuronpedia
from IPython.display import IFrame
html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

def get_dashboard_html(sae_release = "gemma-2-2b", sae_id="20-gemmascope-res-16k", feature_idx=0):
    return html_template.format(sae_release, sae_id, feature_idx)

# f_idx = top_latents[0].item()
f_idx = 83
html = get_dashboard_html(sae_release = "gemma-3-4b-it", sae_id="22-gemmascope-2-res-16k", feature_idx=f_idx)
IFrame(html, width=1200, height=600)
# %%
