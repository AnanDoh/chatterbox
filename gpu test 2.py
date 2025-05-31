import re
import torch
import torchaudio as ta
import os
from chatterbox.tts import ChatterboxTTS

# -----------------------------
# 1) Enhanced text preprocessing for natural pauses
# -----------------------------
def enhance_text_for_natural_speech(text):
    """
    Enhance text with better punctuation for natural pauses and breathing.
    """
    # Add pauses after common sentence starters and transitions
    transition_words = [
        "However", "Therefore", "Meanwhile", "Furthermore", "Nevertheless", 
        "Moreover", "Additionally", "Consequently", "Subsequently", "Finally",
        "First", "Second", "Third", "Next", "Then", "Now", "So", "Well",
        "You know", "I mean", "Actually", "Basically", "Honestly"
    ]
    
    for word in transition_words:
        # Add slight pause after transition words
        text = re.sub(rf'\b{word}\b,?', f'{word},', text, flags=re.IGNORECASE)
    
    # Add pauses for natural breathing points
    text = re.sub(r'(\w+)\s+(and|but|or|yet|so)\s+', r'\1, \2 ', text)
    
    # Enhance dialogue and quotations with pauses
    text = re.sub(r'"([^"]+)"', r', "\1",', text)
    
    # Add pauses before and after parenthetical expressions
    text = re.sub(r'\s*\(([^)]+)\)\s*', r', (\1), ', text)
    
    # Ensure proper pauses after sentence endings
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # Add breathing pauses in long sentences (every 15-20 words)
    sentences = re.split(r'[.!?]+', text)
    enhanced_sentences = []
    
    for sentence in sentences:
        if len(sentence.strip()) == 0:
            continue
            
        words = sentence.strip().split()
        if len(words) > 15:
            # Insert natural pauses in long sentences
            chunks = []
            current_chunk = []
            
            for i, word in enumerate(words):
                current_chunk.append(word)
                
                # Add pause after clauses or at natural break points
                if (len(current_chunk) >= 8 and 
                    (word.endswith(',') or word in ['and', 'but', 'or', 'because', 'when', 'while', 'if', 'that', 'which'])):
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                elif len(current_chunk) >= 15:
                    # Force a pause in very long segments
                    chunks.append(' '.join(current_chunk) + ',')
                    current_chunk = []
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            enhanced_sentences.append(' '.join(chunks))
        else:
            enhanced_sentences.append(sentence.strip())
    
    # Rejoin sentences
    text = '. '.join(enhanced_sentences)
    
    # Clean up multiple commas and spaces
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r',\s*\.', '.', text)
    
    return text.strip()

def chunk_text_at_sentences(text, max_chars=400):
    """
    Enhanced chunking that respects natural speech boundaries and maintains context.
    Reduced chunk size for better processing and more natural pauses between chunks.
    """
    # First enhance the text for natural speech
    text = enhance_text_for_natural_speech(text)
    
    # Split into sentences with better regex
    sentences = re.findall(r'.+?[\.!?]+(?:\s|$)', text, flags=re.S)
    chunks, current = [], ""
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
            
        # If adding this sentence would exceed max_chars
        if len(current) + len(sent) > max_chars:
            if current:
                # Add current chunk and start new one
                chunks.append(current.strip())
                current = sent
            else:
                # Single sentence is too long, split it carefully
                words = sent.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 > max_chars:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            # Single word is too long, just add it
                            chunks.append(word)
                            temp_chunk = ""
                    else:
                        temp_chunk += " " + word if temp_chunk else word
                
                if temp_chunk:
                    current = temp_chunk
                else:
                    current = ""
        else:
            # Add sentence to current chunk
            current += " " + sent if current else sent
    
    # Add final chunk
    if current:
        chunks.append(current.strip())
    
    return chunks

# -----------------------------
# 2) Natural voice parameter configurations
# -----------------------------
class VoicePresets:
    """
    Optimized voice presets based on Chatterbox TTS guide for natural speech.
    """
    
    @staticmethod
    def natural_narration():
        """For calm, natural storytelling - optimized for your long narrative text."""
        return {
            'exaggeration': 0.4,      # Subdued, natural delivery
            'cfg_weight': 0.35,       # Slower, more deliberate pace
            'temperature': 0.6        # Consistent but not robotic
        }
    
    @staticmethod
    def conversational():
        """For dialogue and more expressive parts."""
        return {
            'exaggeration': 0.5,      # Neutral, natural speech
            'cfg_weight': 0.4,        # Moderate pace
            'temperature': 0.7        # Slight variation for naturalness
        }
    
    @staticmethod
    def emotional_moments():
        """For emotionally charged sections."""
        return {
            'exaggeration': 0.6,      # More expressive
            'cfg_weight': 0.3,        # Slower for emotional impact
            'temperature': 0.65       # Controlled variation
        }
    
    @staticmethod
    def reflective():
        """For introspective, thoughtful passages."""
        return {
            'exaggeration': 0.35,     # Very calm
            'cfg_weight': 0.25,       # Very slow, contemplative
            'temperature': 0.55       # Consistent delivery
        }

def detect_text_mood(text):
    """
    Analyze text to determine appropriate voice preset.
    """
    text_lower = text.lower()
    
    # Emotional indicators
    emotional_words = ['cried', 'tears', 'sobbing', 'angry', 'furious', 'excited', 'thrilled', 'shocked', 'devastated']
    dialogue_indicators = ['"', "'", 'said', 'asked', 'replied', 'whispered', 'shouted']
    reflective_words = ['remember', 'thought', 'wondered', 'realized', 'understood', 'reflected', 'considered']
    
    emotional_count = sum(1 for word in emotional_words if word in text_lower)
    dialogue_count = sum(1 for indicator in dialogue_indicators if indicator in text)
    reflective_count = sum(1 for word in reflective_words if word in text_lower)
    
    # Determine mood based on content
    if emotional_count >= 2:
        return VoicePresets.emotional_moments()
    elif dialogue_count >= 3:
        return VoicePresets.conversational()
    elif reflective_count >= 2:
        return VoicePresets.reflective()
    else:
        return VoicePresets.natural_narration()

# -----------------------------
# 3) Set up device (GPU if available)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("Running on CPU")

# Patch torch.load to map to the chosen device
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = device
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

# -----------------------------
# 4) Load model onto GPU
# -----------------------------
print("Loading ChatterboxTTS model...")
model = ChatterboxTTS.from_pretrained(device=device)
print("Model loaded successfully!")

# -----------------------------
# 5) Load text from script.txt if it exists, otherwise use default
# -----------------------------
script_file = "script.txt"
# Define the audio sample file for voice cloning
audio_sample_file = "The Golden Voice - Ted Williams-yt.savetube.me.mp3"

if os.path.exists(script_file):
    print(f"Found {script_file}, using text from file...")
    with open(script_file, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded {len(text)} characters from {script_file}")
else:
    print("No script.txt found, using default text...")
    text = """
You ever walk into a place and swear the walls remember you?

I drove back to Sycamore Hill yesterday. Took the long route through backroads where the trees hang low like old men dozing off in lawn chairs. Nothing's changed, and everything has. The gas station still sells that dusty jerky I never trusted, but now there's a touchscreen soda machine inside. Fancy. I parked in front of the old house, sat there with the engine running like I might need to make a quick getaway from a ghost.

I hadn't been back in seventeen years. Not since Mom passed and Dad packed up what was left of his heart into a storage unit two states away. I didn't come back for the funeral. I couldn't. Didn't know how to grieve a place as much as a person. Grief is strange like that—it wears all kinds of masks. Sometimes it's quiet, like missing the way someone folds towels. Other times it screams in the middle of a grocery store because you saw their favorite cereal.

But there I was, staring at the house like it might blink.

I remember when we moved in. I was seven. Dad painted my room this awful mustard yellow because I said I liked gold once. He tried his best, always did. Mom filled the kitchen with little herb pots—rosemary, basil, mint—and every night smelled like garlic and lavender and warmth. It wasn't perfect, but it was *ours*. Until it wasn't.

Funny how fast "forever" can fall apart.

The porch steps creaked just like I remembered. That third one still threatens to snap your ankle if you don't step just right. I ran my hand along the railing, and I swear the splinter I got when I was twelve is still there waiting. I brought the spare key—they never changed the lock—and I let myself in.

Dust doesn't make a sound, but it has a presence. Like it's watching. Every surface held a memory. The coffee stain on the side table where Dad used to read his paper. The gouge in the floor from when I dropped Mom's cast-iron skillet. I used to lie about that scratch. Told her the cat did it. We didn't have a cat.

The living room felt smaller. That happens, doesn't it? As kids, everything looks big and mysterious. As adults, it's all just drywall and regrets. I sat on the floor, cross-legged like I used to do with my homework. The light through the window hit the spot where Mom's rocking chair used to be. She used to hum there, real soft, like she was keeping the house alive with her voice.

I stayed like that for a while. Didn't check my phone. Didn't speak. Just listened. And maybe I'm crazy, but I *heard* her. Not her voice exactly—more like her presence. The way the room felt fuller for a second. Like she had just stepped into the kitchen and would be right back with a cup of tea.

I walked down the hallway, counted the floorboards out of habit. Fourteen to the bathroom. Sixteen to my old room. I paused at the doorway.

It still had the dent in the wall from when I slammed the door during a fight. Don't even remember what we were arguing about. Probably something dumb. Probably something I wish I could take back. The room was empty now, except for the sunlight crawling across the floor like a memory trying to find its way back to me.

You know, I used to dream big in that room. Astronaut. Writer. Guitar player. None of those happened, really. I'm not complaining—I've made peace with who I became. But part of me still wonders if those dreams got left behind in this room, waiting for me to come back.

I went upstairs next, to the attic. The door stuck the same way it always did—you have to lift then push. My fingers found the motion automatically, like a muscle memory I didn't know I still had. It was hot up there. The kind of heat that makes you sweat out secrets. Boxes lined the far wall, covered in more dust than tape. I picked one at random and opened it.

Photos. Letters. Report cards. Every version of me stared back from those snapshots. Bowl cut, braces, bad fashion choices. One photo had Mom holding me at a fair, cotton candy in one hand, joy in her eyes. I don't remember that day. But the photo remembered for me.

There was a letter she'd written but never sent. Addressed to me. The paper was yellowed, but her handwriting still danced across it like she was whispering every word. She wrote about how proud she was. About how she knew I struggled sometimes to understand my place in the world, but that she always saw the good in me. Said she hoped I'd come back someday. Said the house would wait.

And I cried. Not loud. Not movie-style with sobs and shaking shoulders. Just quiet tears. The kind that sneak out before you notice. I sat there, clutching a letter from a ghost, and felt the years melt away. I wasn't thirty-four in that moment. I was just a kid missing his mom.

When I left the attic, I didn't close the box. I let the light from the window keep the memories company.

Back downstairs, I stood in the kitchen. It smelled like dust and silence, but in my mind, it was garlic and rosemary again. I opened a cabinet and found one of her old mugs. Cracked but still standing. Like me.

I didn't take anything. Didn't need to. I wasn't here to reclaim things—I was here to remember. To forgive. To let go.

When I stepped out the front door, I turned around one last time. The sun hit the windows just right, and for a moment, I swear it looked like the house smiled.

I got in my car, started the engine, and pulled away slowly.

No ghosts chased me. No memories begged to come with me. They stayed where they belonged—inside that house on Sycamore Hill.

And that's okay.

Because now, I carry them in my own way.
"""

# Check if the audio sample file exists
if not os.path.exists(audio_sample_file):
    print(f"Warning: Audio sample file '{audio_sample_file}' not found!")
    print("The TTS will use the default voice instead of voice cloning.")
    audio_sample_file = None
else:
    print(f"Found audio sample: {audio_sample_file}")
    print("Will use this audio for voice cloning.")

# -----------------------------
# 6) Enhanced chunking and synthesis with adaptive voice parameters
# -----------------------------
chunks = chunk_text_at_sentences(text, max_chars=400)  # Smaller chunks for better control
wavs_cpu = []
chunk_files = []

print(f"\nProcessing {len(chunks)} chunks with adaptive voice parameters...")
print("Voice parameters will be automatically adjusted based on content mood.")

# Set a consistent seed for reproducible results (optional)
torch.manual_seed(42)

for idx, chunk in enumerate(chunks, start=1):
    print(f"\nProcessing chunk {idx}/{len(chunks)}...")
    print(f"Text preview: {chunk[:100]}{'...' if len(chunk) > 100 else ''}")
    
    # Detect mood and get appropriate voice parameters
    voice_params = detect_text_mood(chunk)
    print(f"Detected mood - Exaggeration: {voice_params['exaggeration']}, "
          f"CFG Weight: {voice_params['cfg_weight']}, "
          f"Temperature: {voice_params['temperature']}")
    
    # Generate with optimized parameters for natural speech
    try:
        if audio_sample_file:
            wav_gpu = model.generate(
                chunk, 
                audio_prompt_path=audio_sample_file,
                exaggeration=voice_params['exaggeration'],
                cfg_weight=voice_params['cfg_weight'],
                temperature=voice_params['temperature']
            )
        else:
            wav_gpu = model.generate(
                chunk,
                exaggeration=voice_params['exaggeration'],
                cfg_weight=voice_params['cfg_weight'],
                temperature=voice_params['temperature']
            )
        
        # Move to CPU for saving
        wav = wav_gpu.detach().cpu()
        filename = f"chunk_{idx:03d}.wav"
        chunk_files.append(filename)
        ta.save(filename, wav, model.sr)
        print(f"✓ Saved {filename} ({len(chunk)} chars)")
        wavs_cpu.append(wav)
        
    except Exception as e:
        print(f"✗ Error processing chunk {idx}: {e}")
        continue

# -----------------------------
# 7) Stitch chunks with natural pauses between them
# -----------------------------
print("\nStitching chunks together with natural inter-chunk pauses...")

if wavs_cpu:
    # Add small silence between chunks for natural flow (0.3 seconds)
    silence_duration = int(0.3 * model.sr)  # 0.3 seconds of silence
    silence = torch.zeros(1, silence_duration)
    
    # Combine all chunks with pauses
    final_parts = []
    for i, wav in enumerate(wavs_cpu):
        final_parts.append(wav)
        # Add pause between chunks (except after the last one)
        if i < len(wavs_cpu) - 1:
            final_parts.append(silence)
    
    full_wav = torch.cat(final_parts, dim=1)
    ta.save("full_output_natural.wav", full_wav, model.sr)
    print("✓ Saved full_output_natural.wav with enhanced natural speech")
    
    # Also save without pauses for comparison
    full_wav_no_pauses = torch.cat(wavs_cpu, dim=1)
    ta.save("full_output_no_pauses.wav", full_wav_no_pauses, model.sr)
    print("✓ Saved full_output_no_pauses.wav for comparison")
else:
    print("✗ No audio chunks were successfully generated!")

# -----------------------------
# 8) Clean up chunk files (optional)
# -----------------------------
cleanup_chunks = input("\nDelete individual chunk files? (y/N): ").lower().strip()
if cleanup_chunks == 'y':
    print("Cleaning up chunk files...")
    for chunk_file in chunk_files:
        try:
            os.remove(chunk_file)
            print(f"✓ Deleted {chunk_file}")
        except OSError as e:
            print(f"✗ Error deleting {chunk_file}: {e}")
else:
    print("Keeping individual chunk files for review.")

print("\n🎉 Processing complete!")
print("\nGenerated files:")
print("- full_output_natural.wav (with enhanced natural speech and pauses)")
print("- full_output_no_pauses.wav (for comparison)")
if not cleanup_chunks == 'y':
    print(f"- {len(chunk_files)} individual chunk files (chunk_001.wav, etc.)")

print("\n📝 Voice Enhancement Features Applied:")
print("✓ Adaptive voice parameters based on text mood")
print("✓ Enhanced text preprocessing for natural pauses")
print("✓ Optimized chunking for better speech flow")
print("✓ Natural inter-chunk pauses")
print("✓ Mood-based parameter adjustment (narration/emotional/reflective)")
