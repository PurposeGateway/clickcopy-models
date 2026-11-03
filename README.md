# clickcopy-models

ONNX model files, published as release assets (never committed to this repository).

## wav2vec2-large-lv60k-960h-int4.onnx

A CTC character model for English speech: wav2vec 2.0 Large (LV-60k), fine-tuned on 960 h of LibriSpeech,
exported to ONNX with 4-bit weight-only compression of its linear layers. It produces per-frame label logits
for 16 kHz mono audio and is suitable for CTC forced alignment of a known transcript.

| | |
|---|---|
| Release tag | `wav2vec2-large-lv60k-960h-v1` |
| File | `wav2vec2-large-lv60k-960h-int4.onnx` |
| Size | 214090137 bytes |
| SHA-256 | `cba21b099e3cb8ad81a91b72171020ae1e8ea3e6b439d53a4c9078cf7964c824` |
| Download | https://github.com/PurposeGateway/clickcopy-models/releases/download/wav2vec2-large-lv60k-960h-v1/wav2vec2-large-lv60k-960h-int4.onnx |

### Source weights

- torchaudio bundle `torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H` (torchaudio 2.8.0), the fairseq
  wav2vec 2.0 Large LV-60k + 960 h LibriSpeech checkpoint, published under the MIT License (see `LICENSE` and `NOTICE`).
- Checkpoint: https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_large_lv60k_asr_ls960.pth
  - size 1,262,046,195 bytes
  - SHA-256 `7a88965716fbd598a595209bf45c1210a18a6935cfb0cf53527fc986c5543ac7`

### Export settings

- Exported model: the bundle's inner `Wav2Vec2Model` (`bundle.get_model().model`), with `torch.onnx.export`
  (torch 2.8.0, `dynamo=False`), opset 17.
- Input `input_values`: float32, shape `[1, samples]`, 16 kHz mono, samples in [-1, 1].
- Output `emission`: float32, shape `[1, frames, 29]`, raw logits, one frame per 320 samples (20 ms).
- The caller does what the torchaudio bundle's wrapper does around the inner model:
  - **before** the model: layer-normalise the whole input passed in each call, `x = (x - mean(x)) / sqrt(var(x) + 1e-5)` (eps `1e-5`);
  - **after** the model: `log_softmax` over the label axis.
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

### Conversion of this file

- From the fp32 ONNX export above (1,262,324,814 bytes) with ONNX Runtime 1.23.2
  `onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer` and
  `DefaultWeightOnlyQuantConfig(bits=4)`: block size 128, asymmetric, `accuracy_level` unset.
- The 146 `MatMul` nodes with constant weights become `com.microsoft` `MatMulNBits` (4-bit weights, per-block scales
  and zero points). The attention `MatMul`s between activations, the convolutions, layer norms and biases stay fp32.
- Runs on the ONNX Runtime CPU execution provider (checked with onnxruntime 1.23.2 on Windows x64).

## Licence

The weights are MIT-licensed by Facebook, Inc. and its affiliates; see `LICENSE`. The ONNX conversion is ours; see `NOTICE`.
