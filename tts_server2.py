#!/usr/bin/env python3

import os
import re
import socket
import sys
import datetime
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

##############################################################################
# NIEMALS INS INTERNET
##############################################################################

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

##############################################################################
# KONFIGURATION
##############################################################################

PORT = int(sys.argv[1]) if len(sys.argv) >= 2 else 9999
OUTPUT_DIR = Path("/tmp")
SPEAKER_DIR = Path(__file__).resolve().parent / "stimmen"
DEFAULT_REF_AUDIO = "/home/Qwen3-TTS/my_voice.wav"
DEFAULT_REF_TEXT = """
MY REFERENCE TEST FOR THE WAV FILE 
"""

##############################################################################
# HUGGINGFACE CACHE
##############################################################################

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
folder_name = "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"
snapshot_dir = cache_dir / folder_name / "snapshots"

if not snapshot_dir.exists():
    raise FileNotFoundError(f"Kein Snapshot gefunden:\n{snapshot_dir}")

snapshots = sorted(snapshot_dir.iterdir())
if not snapshots:
    raise FileNotFoundError("Keine Modell-Snapshots gefunden.")
model_path = snapshots[-1]

##############################################################################
# MODELL LADEN
##############################################################################

print("Lade Modell ...")
model = Qwen3TTSModel.from_pretrained(
    str(model_path),
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    local_files_only=True,
)

##############################################################################
# HELFER
##############################################################################

SPEAKER_RE = re.compile(r"^\s*\[speaker=([^\]]+)\](.*)$", re.DOTALL)


def parse_speaker(text):
    m = SPEAKER_RE.match(text)
    if not m:
        return None, DEFAULT_REF_AUDIO, DEFAULT_REF_TEXT, text.strip()
    name = m.group(1).strip()
    rest = m.group(2).strip()
    audio = SPEAKER_DIR / f"{name}.wav"
    text_file = SPEAKER_DIR / f"{name}.txt"
    if not audio.is_file():
        raise FileNotFoundError(f"Kein Speaker-Audio: {audio}")
    if not text_file.is_file():
        raise FileNotFoundError(f"Kein Speaker-Text: {text_file}")
    ref_text = text_file.read_text(encoding="utf-8").strip()
    return name, str(audio), ref_text, rest


prompt_cache = {}


def get_clone_prompt(ref_audio, ref_text):
    key = (ref_audio, ref_text)
    if key not in prompt_cache:
        prompt_cache[key] = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
    return prompt_cache[key]


def generate_filename():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUTPUT_DIR / f"ausgabe_{ts}"
    path = OUTPUT_DIR / f"{base}.wav"
    counter = 1
    while path.exists():
        path = OUTPUT_DIR / f"{base}.{counter}.wav"
        counter += 1
    return path


def handle_connection(conn):
    try:
        chunks = []
        while True:
            data = conn.recv(65536)
            if not data:
                break
            chunks.append(data)
        text = b"".join(chunks).decode("utf-8", errors="replace")

        if not text.strip():
            conn.sendall(b"Fehler: Leerer Text.\n")
            return

        name, ref_audio, ref_text, speech = parse_speaker(text)
        if not speech:
            conn.sendall(b"Fehler: Kein zu sprechender Text.\n")
            return

        who = f"Sprecher [{name}] " if name else ""
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {who}Satz: {speech}")

        start = datetime.datetime.now()
        prompt = get_clone_prompt(ref_audio, ref_text)
        audio, sample_rate = model.generate_voice_clone(
            text=speech,
            voice_clone_prompt=prompt,
        )
        elapsed = datetime.datetime.now() - start

        out_path = generate_filename()
        sf.write(out_path, audio[0], sample_rate)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {who}Fertig in {elapsed} -> {out_path}")
        conn.sendall(f"{out_path}\n".encode())
    except FileNotFoundError as e:
        conn.sendall(f"Fehler: {e}\n".encode())
    except Exception as e:
        conn.sendall(f"Fehler: {e}\n".encode())

##############################################################################
# SERVER
##############################################################################

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(5)
    print(f"Lausche auf Port {PORT}. Beenden mit Strg+C.")

    while True:
        try:
            conn, addr = srv.accept()
        except KeyboardInterrupt:
            print("\nServer beendet.")
            break
        with conn:
            print(f"Verbindung von {addr}")
            try:
                handle_connection(conn)
            except KeyboardInterrupt:
                print("\nServer beendet (unterbrochen).")
                break
