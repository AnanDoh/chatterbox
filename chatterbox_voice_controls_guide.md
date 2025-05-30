# Chatterbox TTS Voice Controls Guide 🎙️

## Overview
Chatterbox is Resemble AI's production-grade open source TTS model with unique emotion exaggeration control. This guide covers all available voice controls and parameters.

---

## 🎛️ Core Voice Parameters

### 1. **Exaggeration** 🎭
- **Range**: 0.25 - 2.0
- **Default**: 0.5
- **Purpose**: Controls emotion intensity and expressiveness
- **First open-source TTS with this feature!**

```python
# Neutral, natural speech
exaggeration=0.5

# Subdued, calm delivery
exaggeration=0.25-0.4

# Dramatic, expressive speech
exaggeration=0.7-2.0

# Extreme dramatic (can be unstable)
exaggeration=1.5-2.0
```

### 2. **CFG Weight (Classifier-Free Guidance/Pace)** ⚡
- **Range**: 0.2 - 1.0
- **Default**: 0.5
- **Purpose**: Controls speech pacing and adherence to reference voice

```python
# Very slow, deliberate speech
cfg_weight=0.2-0.3

# Moderate pace
cfg_weight=0.4-0.5

# Faster speech, stronger guidance
cfg_weight=0.6-1.0
```

### 3. **Temperature** 🌡️
- **Range**: 0.05 - 5.0
- **Default**: 0.8
- **Purpose**: Controls randomness and creativity

```python
# Consistent, predictable output
temperature=0.1-0.5

# Balanced creativity
temperature=0.6-1.0

# High creativity/variation
temperature=1.5-5.0
```

### 4. **Audio Prompt/Reference Voice** 🎤
- **Input**: Audio file path or None
- **Formats**: WAV, MP3, FLAC, OPUS
- **Duration**: 3-10 seconds recommended
- **Purpose**: Voice cloning from reference audio

```python
# Use reference voice
audio_prompt_path="path/to/reference.wav"

# Use default voice
audio_prompt_path=None
```

### 5. **Random Seed** 🎲
- **Range**: Any integer
- **Default**: 0 (random)
- **Purpose**: Reproducible results

```python
# Random generation
seed=0

# Reproducible output
seed=42  # or any fixed number
```

---

## 🎯 Preset Configurations

### **Natural Conversation** 💬
```python
model.generate(
    text="Hello, how are you today?",
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8
)
```

### **Slow, Deliberate Speech** 🐌
```python
model.generate(
    text="Let me explain this carefully.",
    exaggeration=0.3,
    cfg_weight=0.2,
    temperature=0.6
)
```

### **Fast, Energetic Speech** ⚡
```python
model.generate(
    text="This is so exciting!",
    exaggeration=0.7,
    cfg_weight=0.7,
    temperature=0.9
)
```

### **Dramatic Performance** 🎭
```python
model.generate(
    text="To be or not to be, that is the question.",
    exaggeration=1.0,
    cfg_weight=0.3,
    temperature=0.8
)
```

### **Calm Narration** 📖
```python
model.generate(
    text="Once upon a time in a distant land...",
    exaggeration=0.4,
    cfg_weight=0.4,
    temperature=0.6
)
```

### **Consistent Output** 🔒
```python
model.generate(
    text="This will sound the same every time.",
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.3,
    seed=42
)
```

### **High Creativity** 🎨
```python
model.generate(
    text="Let's try something different!",
    exaggeration=0.6,
    cfg_weight=0.4,
    temperature=1.5
)
```

---

## 🔄 Voice Conversion Options

### **Basic Voice Conversion**
```python
from chatterbox.vc import ChatterboxVC

model = ChatterboxVC.from_pretrained(device)
wav = model.generate(
    source_audio_path,
    target_voice_path=target_speaker_path
)
```

### **Command Line Voice Conversion**
```bash
python voice_conversion.py input.wav target_speaker.wav -o output_folder
python voice_conversion.py input_folder/ target_speaker.wav -o output_folder
```

**Options:**
- `-o, --output_folder`: Output directory
- `-g, --gpu_id`: GPU ID to use
- `--no-watermark`: Skip watermarking

---

## 🎚️ Advanced Parameter Combinations

### **For Fast-Speaking Reference Voices**
```python
# README recommendation
model.generate(
    text="Your text here",
    audio_prompt_path="fast_speaker.wav",
    exaggeration=0.5,
    cfg_weight=0.3,  # Lower to compensate for fast reference
    temperature=0.8
)
```

### **Expressive + Controlled Pace**
```python
# High emotion but controlled speed
model.generate(
    text="I can't believe this happened!",
    exaggeration=0.8,
    cfg_weight=0.3,  # Slow down the high emotion
    temperature=0.7
)
```

### **Subtle Variations**
```python
# Small adjustments for fine-tuning
model.generate(
    text="Fine-tuned speech output",
    exaggeration=0.55,  # Slightly more expressive
    cfg_weight=0.45,    # Slightly slower
    temperature=0.75    # Slightly less random
)
```

---

## 🛠️ Technical Configuration

### **Model Loading Options**
```python
# GPU (recommended)
model = ChatterboxTTS.from_pretrained(device="cuda")

# CPU (slower)
model = ChatterboxTTS.from_pretrained(device="cpu")

# Specific GPU
model = ChatterboxTTS.from_pretrained(device="cuda:0")
```

### **Audio Processing Settings**
```python
# Sample rate (read-only)
print(model.sr)  # Usually 24000 Hz

# Save generated audio
import torchaudio as ta
ta.save("output.wav", wav, model.sr)
```

### **Memory Management**
```python
# For long texts or batch processing
with torch.inference_mode():
    wav = model.generate(text, ...)
```

---

## 🎪 Creative Use Cases

### **Character Voices**
```python
# Wise old character
exaggeration=0.3, cfg_weight=0.2, temperature=0.6

# Excited child character  
exaggeration=0.8, cfg_weight=0.6, temperature=1.0

# Mysterious narrator
exaggeration=0.4, cfg_weight=0.3, temperature=0.5

# Energetic announcer
exaggeration=0.7, cfg_weight=0.7, temperature=0.9
```

### **Content Types**
```python
# Podcast narration
exaggeration=0.4, cfg_weight=0.4, temperature=0.7

# Audiobook reading
exaggeration=0.3, cfg_weight=0.3, temperature=0.5

# Video game dialogue
exaggeration=0.6, cfg_weight=0.5, temperature=0.8

# Educational content
exaggeration=0.4, cfg_weight=0.4, temperature=0.6

# Marketing/commercial
exaggeration=0.6, cfg_weight=0.6, temperature=0.8
```

---

## 🚨 Troubleshooting Tips

### **Speech Too Fast**
- Lower `cfg_weight` (0.2-0.3)
- Lower `exaggeration` (0.3-0.4)
- Lower `temperature` (0.5-0.7)

### **Speech Too Slow**
- Raise `cfg_weight` (0.6-0.8)
- Raise `exaggeration` (0.6-0.8)
- Check reference audio pace

### **Inconsistent Output**
- Lower `temperature` (0.3-0.6)
- Set fixed `seed` value
- Use more stable parameter ranges

### **Unnatural Sounding**
- Adjust `exaggeration` closer to 0.5
- Try different reference audio
- Balance `cfg_weight` around 0.4-0.6

### **Extreme Values Warning**
- `exaggeration > 1.5` can be unstable
- `temperature > 2.0` may produce artifacts
- `cfg_weight < 0.2` may be too slow

---

## 📊 Parameter Interaction Matrix

| Exaggeration | CFG Weight | Result |
|--------------|------------|---------|
| Low (0.3) | Low (0.3) | Calm, slow speech |
| Low (0.3) | High (0.7) | Calm but faster |
| High (0.8) | Low (0.3) | Dramatic but controlled |
| High (0.8) | High (0.7) | Very expressive and fast |

---

## 🔧 Gradio Interface Controls

When using `gradio_tts_app.py`:

1. **Text Input**: Enter text to synthesize
2. **Reference Audio**: Upload file or record via microphone
3. **Exaggeration Slider**: 0.25-2.0 range
4. **CFG/Pace Slider**: 0.2-1.0 range
5. **More Options (Accordion)**:
   - Random seed input
   - Temperature slider: 0.05-5.0

---

## 📝 Best Practices

1. **Start with defaults** and adjust incrementally
2. **Test with short phrases** before long texts
3. **Use appropriate reference audio** (clear, 3-10 seconds)
4. **Balance exaggeration and cfg_weight** for desired effect
5. **Set seeds for reproducible results** in production
6. **Monitor for extreme values** that may cause instability
7. **Consider content type** when choosing parameters

---

## 🎵 Audio Quality Notes

- **Built-in watermarking**: All outputs include Perth watermarks
- **Sample rate**: 24kHz output
- **Formats supported**: Input (WAV, MP3, FLAC, OPUS), Output (WAV)
- **Duration limits**: No hard limits, but longer texts may need chunking
- **Quality factors**: Clean reference audio improves results

---

*This guide covers Chatterbox TTS v1.0. For updates and advanced features, check the official repository.* 