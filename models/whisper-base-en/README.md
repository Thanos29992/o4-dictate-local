---
license: apache-2.0
license_link: https://choosealicense.com/licenses/apache-2.0/
language: 
- en
tags:
- audio
- automatic-speech-recognition
- hf-asr-leaderboard
---

# whisper-base.en-int4-ov
* Model creator: [OpenAI](https://huggingface.co/openai)
 * Original model: [whisper-base.en](https://huggingface.co/openai/whisper-base.en)

## Description
This is [whisper-base.en](https://huggingface.co/openai/whisper-base.en) model converted to the [OpenVINO™ IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) (Intermediate Representation) format with weights compressed to INT4 by [NNCF](https://github.com/openvinotoolkit/nncf).


## Quantization Parameters

Weight compression was performed using `nncf.compress_weights` with the following parameters:

* mode: **INT4_ASYM**
* ratio: **1.0**
* group_size: **128**

For more information on quantization, check the [OpenVINO model optimization guide](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html).


## Compatibility

The provided OpenVINO™ IR model is compatible with:

* OpenVINO version 2025.2.0 and higher
* Optimum Intel 1.23.0 and higher


## Running Model Inference with [Optimum Intel](https://huggingface.co/docs/optimum/intel/index)

1. Install packages required for using [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) integration with the OpenVINO backend:

```
pip install optimum[openvino] "datasets<4" librosa soundfile --extra-index-url https://download.pytorch.org/whl/cpu
```

2. Run model inference:

```
from datasets import load_dataset
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForSpeechSeq2Seq

model_id = "OpenVINO/whisper-base.en-int4-ov"
tokenizer = AutoProcessor.from_pretrained(model_id)
model = OVModelForSpeechSeq2Seq.from_pretrained(model_id)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
sample = dataset[0]

input_features = tokenizer(
    sample["audio"]["array"],
    sampling_rate=sample["audio"]["sampling_rate"],
    return_tensors="pt",
).input_features

outputs = model.generate(input_features)
text = tokenizer.batch_decode(outputs)[0]
print(text)
```

## Running Model Inference with [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)

1. Install packages required for using OpenVINO GenAI.
```
pip install huggingface_hub "datasets<4" librosa soundfile
pip install -U --pre --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly openvino openvino-tokenizers openvino-genai
```

2. Download model from HuggingFace Hub
   
```
import huggingface_hub as hf_hub

model_id = "OpenVINO/whisper-base.en-int4-ov"
model_path = "whisper-base.en-int4-ov"

hf_hub.snapshot_download(model_id, local_dir=model_path)

```

3. Run model inference:

```
import openvino_genai as ov_genai
import datasets

device = "CPU"
pipe = ov_genai.WhisperPipeline(model_path, device)

dataset = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
sample = dataset[0]["audio"]["array"]
print(pipe.generate(sample))
```

More GenAI usage examples can be found in OpenVINO GenAI library [docs](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md) and [samples](https://github.com/openvinotoolkit/openvino.genai?tab=readme-ov-file#openvino-genai-samples)

## Limitations

Check the original model card for [original model card](https://huggingface.co/openai/whisper-base.en) for limitations.

## Legal information

The original model is distributed under [apache-2.0](https://choosealicense.com/licenses/apache-2.0/) license. More details can be found in [original model card](https://huggingface.co/openai/whisper-base.en).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel’s Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf). Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.
