# QWEN3TTS

Hi,

you need to install the qwen3 model yourself first:

https://github.com/QwenLM/Qwen3-TTS#environment-setup

this version of a tts focuses on your own voice for speaking, not the build-in ones. So you will need this:

```
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./Qwen3-TTS-12Hz-0.6B-Base
```

you may need to adjust the path the py file searches for the model, according to the local_dir you use in the end.

the subfolder "stimmen" (aka voices) stores .wav / .txt pairs of reference audio and text. 

Start it via "python tts_server2.py" and then connect to it via netcat or the qwen3tts script from PVA Repo.

have fun.
