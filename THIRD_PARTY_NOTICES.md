# Third-party notices

AI Article Check downloads and uses the following model at runtime. Model
weights are not included in this repository archive.

## TMR AI Text Detector

- Original model: [Oxidane/tmr-ai-text-detector](https://huggingface.co/Oxidane/tmr-ai-text-detector)
- ONNX conversion: [onnx-community/tmr-ai-text-detector-ONNX](https://huggingface.co/onnx-community/tmr-ai-text-detector-ONNX)
- License: MIT

The model is an English RoBERTa-based AI-text classifier trained on the RAID
dataset. AI Article Check uses the quantized `onnx/model_int8.onnx` artifact.
The model's output remains probabilistic and can be wrong.

## Optional evaluation candidates

The model-comparison command may download the following weights for local
evaluation. They are not activated automatically and are not included in this
repository archive.

### GLYPH v1.1

- Model: [ogmatrixllm/glyph-v1.1](https://huggingface.co/ogmatrixllm/glyph-v1.1)
- License: MIT
- Evaluated artifact: `onnx_model_quantized.onnx`

GLYPH is an English DeBERTa-v3-base AI-text classifier. The comparison registry
pins the exact model commit and uses the slow SentencePiece tokenizer required
by its model card.

### Fakespot RoBERTa-base AI Text Detection v1

- Original model: [fakespot-ai/roberta-base-ai-text-detection-v1](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1)
- ONNX conversion: [Lynote/fakespot-ai-roberta-base-ai-text-detection-v1-browser](https://huggingface.co/Lynote/fakespot-ai-roberta-base-ai-text-detection-v1-browser)
- License: Apache-2.0
- Evaluated artifact: `onnx/model.onnx`

The ONNX repository describes itself as an unmodified format conversion of the
upstream Fakespot weights. The comparison registry pins the exact conversion
commit.

## Human ChatGPT Comparison Corpus (HC3)

- Dataset: [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
- Paper: *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection*
- License: CC-BY-SA-4.0, subject to stricter licenses of original source data

Benchmark text is downloaded only when the benchmark builder is run and is not
included in this repository archive. The generated metadata records the pinned
source revision and dataset digest.

## MAGE: Machine-generated Text Detection in the Wild

- Dataset: [yaful/MAGE](https://huggingface.co/datasets/yaful/MAGE)
- Paper: *MAGE: Machine-generated Text Detection in the Wild* (ACL 2024)
- License: Apache-2.0

The project downloads the pinned official MAGE validation CSV for length-aware
calibration and the separately pinned test CSV for external evaluation. Test
records never set a calibration parameter or threshold. Source texts, generated
benchmark files, and raw score caches are not included in this repository archive.
