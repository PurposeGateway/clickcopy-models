# clickcopy-models

ONNX model files, published as release assets (never committed to this repository).

## wav2vec2-base-960h.onnx

A CTC character model for English speech: wav2vec 2.0 Base, fine-tuned on 960 h of LibriSpeech, exported to
ONNX in fp32. It produces per-frame label logits for 16 kHz mono audio and is suitable for CTC forced alignment
of a known transcript.

| | |
|---|---|
| Release tag | `wav2vec2-base-960h-v1` |
| File | `wav2vec2-base-960h.onnx` |
| Size | 377811069 bytes |
| SHA-256 | `18faee93850484e85437cd095e53162f2e4f33925398630935fac5730c2f58d4` |
| Download | https://github.com/PurposeGateway/clickcopy-models/releases/download/wav2vec2-base-960h-v1/wav2vec2-base-960h.onnx |

### Source weights

- torchaudio bundle `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H` (torchaudio 2.8.0), the fairseq wav2vec 2.0 Base
  + 960 h LibriSpeech checkpoint, published under the MIT License (see `LICENSE` and `NOTICE`).
- Checkpoint: https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth
  - size 377664473 bytes
  - SHA-256 `488fd4f16de84438ffc945334278c1b9fb9b7159a806c1080b16111a958c945d`

### Export settings

- Exported model: the bundle's `Wav2Vec2Model` (`bundle.get_model()`, which for this bundle has no wrapper), with
  `torch.onnx.export` (torch 2.8.0, `dynamo=False`), opset 17, all weights fp32, no quantisation.
- Input `input_values`: float32, shape `[1, samples]`, 16 kHz mono.
- Output `emission`: float32, shape `[1, frames, 29]`, raw logits, one frame per 320 samples (20 ms).
- The caller must:
  - **before** the model: normalise each input passed to the model to zero mean and unit variance,
    `x = (x - mean(x)) / sqrt(var(x) + 1e-7)` (eps `1e-7`), with `x` the waveform in [-1, 1];
  - **after** the model: apply `log_softmax` over the label axis.
- Export check: on 17 s of speech, without normalisation on either side, the ONNX output matches the torchaudio
  model's to a maximum absolute difference of 0.0033 in log-probability, with the same best label in every frame.
- Labels: upper case; blank is index 0 (`-`); `|` is the word separator. In order:

| Index | Label |
|---|---|
| 0 | `-` |
| 1 | `\|` |
| 2 | `E` |
| 3 | `T` |
| 4 | `A` |
| 5 | `O` |
| 6 | `N` |
| 7 | `I` |
| 8 | `H` |
| 9 | `S` |
| 10 | `R` |
| 11 | `D` |
| 12 | `L` |
| 13 | `U` |
| 14 | `M` |
| 15 | `W` |
| 16 | `C` |
| 17 | `F` |
| 18 | `G` |
| 19 | `Y` |
| 20 | `P` |
| 21 | `B` |
| 22 | `V` |
| 23 | `K` |
| 24 | `'` |
| 25 | `X` |
| 26 | `J` |
| 27 | `Q` |
| 28 | `Z` |

## Also published

An alternative under evaluation: release `wav2vec2-large-lv60k-960h-v1` (wav2vec 2.0 Large LV-60k, 4-bit weights); its settings are in the README at that tag.

## Licence

The weights are MIT-licensed by Facebook, Inc. and its affiliates; see `LICENSE`. The ONNX conversion is ours; see `NOTICE`.
