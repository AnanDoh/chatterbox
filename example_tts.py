import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Patch torch.load to automatically map CUDA tensors to CPU
device = "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "I never thought I’d be the one to leave. But this morning, I packed a bag, walked past the coffee still warm on the counter, and didn’t look back. I don’t know where I’m going, just that I need to breathe without asking permission. Maybe I’ll come back, maybe I won’t. All I know is, for the first time in years, the road feels more like home than the house ever did. And that has to mean something… doesn’t it?"
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

