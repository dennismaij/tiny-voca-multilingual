# tiny-voca (Multilingual Edition)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-orange.svg)](https://pytorch.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime%20Web%20%26%20Native-blueviolet.svg)](https://onnxruntime.ai)

**tiny-voca** is an ultra-compact neural text-to-speech (TTS) model designed to run anywhere — on standard CPUs, mobile devices, and directly inside web browsers at **70x real-time**.

This multilingual edition extends the original architecture to support modular, on-demand language activation, introducing a **studio-grade Dutch (`nl`) model** trained from scratch to 100,000 steps on Apple Silicon GPU (MPS), as well as **British English (`en-gb`)** and the original **American English (`en-us`)**.

---

## Highlights

- **Ultra-Compact**: 4.5M parameters (~18 MB PyTorch checkpoint, ~21 MB standalone ONNX model with embedded vocoder).
- **Blazing Fast**: Runs 70x faster than real-time on 4 CPU threads; instantaneous on Apple Silicon MPS and WebAssembly.
- **Multilingual Support**:
  - 🇳🇱 **Dutch (`nl`)**: Trained to 100,000 steps with broadcast neural teacher datasets (Colette `F2` & Maarten `M2`). Warm, natural prosody with sub-0.7 semitone pitch accuracy.
  - 🇬🇧 **British English (`en-gb`)**: Pipeline and Wav2Vec2 CTC phoneme alignment ready (Libby `F2` & Ryan `M2`).
  - 🇺🇸 **American English (`en-us`)**: Original weights and phoneme frontend.
- **Modular Runtime**: Only activates the selected language dictionary and phonemizer in memory.
- **Runs Everywhere**: PyTorch, native ONNX Runtime, and browser-based WebAssembly (`onnxruntime-web`).

---

## Installation

```bash
git clone https://github.com/dennismaij/tiny-voca-multilingual.git
cd tiny-voca-multilingual

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[own]"
pip install edge-tts soundfile onnxruntime
```

---

## Quickstart CLI (`scripts/run_tts.py`)

Synthesize speech directly from the command line using the modular runtime:

### 🇳🇱 Dutch (PyTorch on Apple Silicon MPS or CPU)
```bash
python scripts/run_tts.py \
  --lang nl \
  --checkpoint checkpoints/dutch_studio/inference.pt \
  --voice F2 \
  --text "Hallo allemaal! Welkom bij de Nederlandstalige tiny-voca synthese." \
  --output output_nl_female.wav
```

### 🇳🇱 Dutch Male Voice (`M2`)
```bash
python scripts/run_tts.py \
  --lang nl \
  --checkpoint checkpoints/dutch_studio/inference.pt \
  --voice M2 \
  --text "De klank is warm, helder en natuurlijk." \
  --output output_nl_male.wav
```

### ⚡ Standalone ONNX Runtime (Zero-PyTorch Deployment)

The pre-trained Dutch model is bundled directly in the repository under [`exported/tiny_voca_nl.onnx`](exported/tiny_voca_nl.onnx) (fine-tuned on tech loanwords and morphological stress rules):

```bash
python scripts/run_tts.py \
  --lang nl \
  --onnx exported/tiny_voca_nl.onnx \
  --voice F2 \
  --text "Dit model draait razendsnel met ONNX Runtime zonder PyTorch!" \
  --output output_onnx.wav
```

Audio samples are available in the [`samples/`](samples/) directory:
- [`samples/umbris_finetuned_f2.wav`](samples/umbris_finetuned_f2.wav) (Female - Colette)
- [`samples/umbris_finetuned_m2.wav`](samples/umbris_finetuned_m2.wav) (Male - Maarten)

---

## Python API

```python
import tiny_voca

# 1. Initialize pipeline for the desired language
pipeline = tiny_voca.get_pipeline(
    language="nl",
    checkpoint="checkpoints/dutch_studio/inference.pt",
    device="mps"  # or "cpu", "cuda"
)

# 2. Synthesize to audio file
pipeline.speak_to_file(
    text="Het weer in Nederland is vandaag prachtig.",
    output_path="weerbericht.wav",
    voice="F2",
    speed=1.0
)

# 3. Or obtain raw float32 waveform and sample rate (24 kHz)
waveform, sample_rate = pipeline.synthesize("Tot ziens!", voice="M2")
```

---

## Exporting to ONNX

To generate a standalone `.onnx` file and metadata bundle for browser or edge deployment:

```bash
python -m tiny_voca.export \
  --checkpoint checkpoints/dutch_studio/inference.pt \
  --output exported/tiny_voca_nl.onnx \
  --language nl
```

---

## Training & Dataset Preparation

This repository includes the complete pipeline to build studio datasets, extract acoustic features, perform forced CTC phoneme alignment, and train tiny-voca models from scratch on Apple Silicon GPU (`mps`) or NVIDIA CUDA:

1. **Dataset Generation**:
   ```bash
   python scripts/generate_studio_audio.py --language nl --total 9000
   ```
2. **Feature Extraction**:
   ```bash
   python -m tiny_voca.prepare --inputs datasets/dutch_studio --output features/dutch_studio
   ```
3. **Wav2Vec2 Forced Phoneme Alignment**:
   ```bash
   python -m tiny_voca.annotate \
     --features features/dutch_studio \
     --output annotations/dutch_studio \
     --language nl \
     --device mps
   ```
4. **Adversarial (GAN) Training**:
   ```bash
   python -m tiny_voca.train \
     --language nl \
     --train annotations/dutch_studio/train.jsonl \
     --validation annotations/dutch_studio/validation.jsonl \
     --output checkpoints/dutch_studio \
     --steps 100000 \
     --device mps
   ```

---

## License & Credits

- Based on the original [tiny-voca](https://github.com/n4ze3m/tiny-voca) by [@n4ze3m](https://github.com/n4ze3m).
- Multilingual architecture, Dutch studio model, and Apple Silicon training pipeline by [@dennismaij](https://github.com/dennismaij).
- Released under the [MIT License](LICENSE).

Languages and voices:

- [apptor-tiny-tts](https://huggingface.co/apptor/apptor-tiny-tts): German.
- [tiny-voca-it](https://github.com/carbogninalberto/tiny-voca-it): Italian.
- [zerre-tts](https://github.com/erkinemreta1/zerre-tts): Turkish.
- [korean-tts](https://github.com/ooodev/korean-tts): Korean, and Korean mixed with English.
- [slyszeglosy](https://github.com/damaon/slyszeglosy): Polish.
- [mini-voz](https://github.com/ElMariones/mini-voz): Spanish, with a Castilian accent.
- [tiny-voca-es](https://github.com/stdevMac/tiny-voca): Spanish from Spain, with a browser demo.
- [tiny-voca-fr](https://github.com/DanielSuncost/tiny-voca-fr): French, with a browser demo.
- [tiny-voca-ptbr](https://github.com/SemFence/tiny-voca-ptbr): Brazilian Portuguese, with its own phonemizer.
- [tiny-voca Arabic](https://github.com/BenkadourKhaled/tiny-voca/tree/arabic-halabi-f2): Modern Standard Arabic, trained on a CPU from the Arabic Speech Corpus.
- [Bengali Voice](https://tts.hansda.in/): Bengali script and Romanized Bengali, with a browser demo.
- [alpa](https://huggingface.co/77ethers/alpa): Sanskrit, fine-tuned from tiny-voca on a CPU.
- [fiato](https://huggingface.co/cfahlgren1/fiato): English, with sighs, exhales and gasps.

Ports and apps:

- [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal/tts.md#tiny-voca): runs tiny-voca with `llama-tts`, from [GGUF files](https://huggingface.co/ggml-org/tiny-voca-GGUF).
- [voca](https://github.com/sultson/voca): a rewrite in Rust.
- [tiny-voca-esp32s3](https://github.com/ngxson/tiny-voca-esp32s3): runs on an ESP32-S3 microcontroller.
- [tiny-voca-dashboard](https://github.com/kleenpulse/tiny-voca-dashboard): a local dashboard.
- [tinyvoca](https://github.com/spurious-cox/tinyvoca): a Mac app.
- [tiny-voca-esp32s3](https://github.com/ngxson/tiny-voca-esp32s3): ESP32-S3 port.

Same idea, other sounds:

- [tinygroove](https://github.com/totomanov/tinygroove): a drum machine with no samples.

Made something? Open an issue and I will add it.

## Cost

The whole project cost $9.56 of GPU time on one rented RTX 4090. That is the
dataset, the failed tries and the final training run.

![Cost breakdown](docs/cost.png)

## Support

It cost less than a pizza to make. If you like it, you can buy me one:
https://ko-fi.com/n4ze3m

## License

Apache-2.0 for code and weights. CC BY 4.0 for the dataset.

## Citation

```bibtex
@software{nazeem2026tinyvoca,
  author = {Nazeem, Muhammed},
  title  = {tiny-voca: a 4.5M-parameter two-voice English text-to-speech model},
  year   = {2026},
  url    = {https://huggingface.co/n4ze3m/tiny-voca}
}
```
