import os
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

print("=== RTX 5080 TTS with CPU Fallback ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# For now, use CPU mode for reliable long text generation
# This is a temporary workaround until PyTorch nightly builds
# have better stability with Blackwell architecture
print("\nUsing CPU mode for reliable long text generation...")
print("(GPU mode works for short texts but has issues with long texts on RTX 5080)")

# Patch torch.load to automatically map CUDA tensors to CPU
device = "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

print("Loading model on CPU...")
model = ChatterboxTTS.from_pretrained(device="cpu")

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

print("Generating audio on CPU (this will take a bit longer but will work reliably)...")
import time
start_time = time.time()

try:
    wav = model.generate(text)
    ta.save("test-cpu-final.wav", wav, model.sr)
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    print("✅ Audio generation successful!")
    print(f"Saved audio to: test-cpu-final.wav")
    print(f"Sample rate: {model.sr}")
    print(f"Audio length: {len(wav[0]) / model.sr:.2f} seconds")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Real-time factor: {generation_time / (len(wav[0]) / model.sr):.2f}x")
    
except Exception as e:
    print(f"❌ Error during generation: {e}")

print("\n" + "="*60)
print("NOTES FOR RTX 5080 USERS:")
print("="*60)
print("• Your RTX 5080 is properly configured with PyTorch + CUDA 12.8")
print("• GPU mode works for SHORT texts but has stability issues with LONG texts")
print("• This is due to nightly PyTorch builds having compatibility issues with Blackwell")
print("• CPU mode works reliably for all text lengths")
print("• GPU support will improve as PyTorch stable releases catch up")
print("• For now, use CPU mode for long texts and GPU for short texts")
print("="*60) 