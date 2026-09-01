# AI Agent–Driven Vision–LLM Diagnostic System for Electronic Components

An end-to-end, AI Agent–driven diagnostic system that integrates MobileNetV3-based visual recognition, C++ ONNX Runtime inference, a QLoRA-fine-tuned Qwen3 LLM, and LangGraph orchestration to identify electronic components, generate structured diagnostic guidance, and route low-confidence cases to human review.
> **Note:** Large Qwen3 GGUF and adapter artifacts are available in the [GitHub Release — artifacts-v1](https://github.com/ChenKC863/AI-Visual-Recognition-and-LLM-Intelligent-Diagnostic-System-for-Electronic-Components/releases).

---

## Dataset

### Sources
- [1] [Electronic-Components-Classification (GitHub)](https://github.com/pouria-faraj/Electronic-Components-Classification)
- [2] [Electronic Components and Devices (Kaggle)](https://www.kaggle.com/datasets/aryaminus/electronic-components)


### Original Dataset(for my choice)

```text
Electronic components/
└── images/
  ├── Inductor/ # 265 images [1]
  ├── Resistor/ # 470 images [1]
  ├── Solenoid/ # 317 images [2]
  └── Transformer/ # 747 images [1]
```

### Cleaned Dataset
After data cleaning and preprocessing:

```text
Electronic components/
└── images/
  ├── Inductor/ # 260 images
  ├── Resistor/ # 470 images
  ├── Solenoid/ # 310 images
  └── Transformer/ # 740 images
```


---
## 📌 Overview

This project presents an end-to-end AI Agent–driven diagnostic system for electronic components, integrating computer vision, C++ ONNX inference, a domain-fine-tuned Large Language Model (LLM), local knowledge retrieval, and LangGraph-based human-in-the-loop orchestration.

The system classifies component images, evaluates prediction confidence, retrieves verified component knowledge, generates structured diagnostic guidance, and conditionally routes low-confidence cases to human review.

The current prototype supports four component classes:

- Inductor
- Resistor
- Solenoid
- Transformer

A balanced working dataset of **1,040 images** was used, with **260 images per class**:

- Training: 728 images
- Validation: 156 images
- Testing: 156 images

### System Workflow

<img width="2575" height="1728" alt="langchain-langgraph-diagnostic-workflow" src="https://github.com/user-attachments/assets/11799058-dc9b-469a-b873-14de81c14f9b" />

The project covers the complete AI lifecycle:

#### 1.) Image classification
MobileNetV3-Small and MobileNetV3-Large models are trained using PyTorch transfer learning. Training uses on-the-fly data augmentation, label smoothing, weight decay, learning-rate scheduling, early stopping, and a two-stage fine-tuning strategy.

#### 2.) Portable C++ inference
The trained vision models are exported to ONNX and executed locally through a C++17 inference engine using ONNX Runtime and OpenCV. The engine performs image preprocessing, classification, latency measurement, confidence visualization, structured JSON output, and batch CSV export.

#### 3.) Domain-specific LLM fine-tuning
Vision predictions are converted into conversational JSONL datasets and used to fine-tune Qwen3-4B-Instruct-2507 with QLoRA, Unsloth, PEFT, and TRL SFTTrainer. The fine-tuned model is exported to GGUF Q4_K_M format and deployed locally through Ollama.

#### 4.) AI Agent orchestration
LangGraph StateGraph orchestrates vision classification, verified knowledge retrieval, confidence-based routing, human-review interrupts, local LLM inference, and deterministic fallback behavior.

#### 5.) Confidence-based human review
A fixed confidence threshold of 0.70 determines whether a prediction can proceed automatically. Predictions below the threshold are routed to human review, where the operator can approve, reject, or correct the predicted component.

#### 6.) Structured diagnostic generation
The local LLM generates structured diagnostic JSON containing the predicted component, vision confidence, review requirement, component function, recommended operator action, and system limitations.

#### 7.) Reliable fallback and policy enforcement
A verified local component knowledge base supports both LLM prompting and deterministic fallback. If the LLM times out or returns invalid JSON, the system constructs a fallback diagnostic response. Final policy normalization prevents the LLM from overriding confidence and human-review rules.


## Project Structure
```text
Four_electronic_components/
├── .git/
├── .gitignore
├── .vscode/
├── .venv/
├── .venv_win/
├── agent/
├── build/
├── configs/
├── knowledge/
├── model/
├── qwen3-4b-instruct-2507-{variant}_model-artifacts/
│     ├── adapter/
│     ├── evaluation/
│     ├── gguf_gguf/
│     ├── prepared_data/
│     ├── artifact_manifest.json
│     ├── confidence_threshold.json
│     ├── environment.json
│     ├── requirements-cloud-lock.txt
│     ├── SHA256SUMS.txt
│     └── training_config.json
├── release_assets/
├── scripts/
├── test/
├── train/
├── val/
├── CMakeLists.txt
├── main.cpp
├── README.md
├── LICENSE
├── requirements-cloud-lock.txt
├── ai-visual-recognition-and-classification-system.ipynb
├── electronics_train_{variant}_model.jsonl
├── electronics_validation_{variant}_model.jsonl
├── electronics_test_{variant}_model.jsonl
└── vision_predictions_{variant}_model.csv
```
where {variant} is small or large.

---


## Getting Started

Here, this guide demonstrates local deployment using **Windows, MSYS2 UCRT64, Ninja, ONNX Runtime, OpenCV, and Ollama**.

### 1. Prerequisites

Install the following software before building the project:

* Git
* Python 3.10 or later
* MSYS2 with the UCRT64 environment
* CMake
* Ninja
* OpenCV
* ONNX Runtime C++ SDK
* Ollama

The GGUF models require several gigabytes of storage. CPU-only execution is supported, but LLM generation may take longer than GPU-accelerated inference.

### 2. Clone the Repository

Open an MSYS2 UCRT64 terminal and run:

```bash
git clone https://github.com/ChenKC863/AI-Visual-Recognition-and-LLM-Intelligent-Diagnostic-System-for-Electronic-Components.git

cd Four_electronic_components
```

### 3. Install the C++ Build Dependencies

Use the **MSYS2 UCRT64** terminal for the C++ workflow.

> If MSYS2 was installed through the CLANG64 package/environment, that is fine. However, the commands below must be executed from **MSYS2 UCRT64**, because this project uses the UCRT64 package prefix, compiler, OpenCV path, and runtime DLLs.

Install the required UCRT64 packages:

```bash
pacman -Syu

pacman -S --needed \
  mingw-w64-ucrt-x86_64-gcc \
  mingw-w64-ucrt-x86_64-cmake \
  mingw-w64-ucrt-x86_64-ninja \
  mingw-w64-ucrt-x86_64-opencv \
  mingw-w64-ucrt-x86_64-curl \
  mingw-w64-ucrt-x86_64-nlohmann-json
```

Confirm that the UCRT64 tools are being used:

```bash
which c++
which cmake
which ninja
```

The paths should point to `/ucrt64/bin/...`.

Download the Windows x64 ONNX Runtime C++ SDK from the official [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) and extract it to a local directory, for example:

```text
C:/onnxruntime/onnxruntime-win-x64-1.27.0/
├── include/
└── lib/
```

Confirm that the extracted package contains the required files:

```text
include/onnxruntime_cxx_api.h
lib/onnxruntime.lib
lib/onnxruntime.dll
```

> The ONNX Runtime version and installation path must match the value supplied to CMake or configured in `CMakeLists.txt`.

### 4. Prepare the Vision Models

Place the exported MobileNetV3 ONNX models in the `model/` directory:

```text
model/
├── component_classifier_small.onnx
├── component_classifier_small.onnx.data
├── component_classifier_large.onnx
└── component_classifier_large.onnx.data
```

If a model was exported without external tensor data, the corresponding `.onnx.data` file may not be required.

The `.onnx` file and its `.onnx.data` file, when present, must remain in the same directory.

## Stage 1

### 5-1. Configure and Build the C++ Inference Engine

From the project root, configure the project with Ninja in the **MSYS2 UCRT64** terminal:

```bash
rm -rf build

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT="C:/onnxruntime/onnxruntime-win-x64-1.27.0" \
  -DOpenCV_DIR="/ucrt64/lib/cmake/opencv4"
```

Replace `C:/onnxruntime/onnxruntime-win-x64-1.27.0` with the actual ONNX Runtime folder on your machine. For example, if the SDK is extracted to `my_project_folder/onnxruntime-win-x64-1.27.0`, use:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT="my_project_folder/onnxruntime-win-x64-1.27.0" \
  -DOpenCV_DIR="/ucrt64/lib/cmake/opencv4"
```

Build the executable:

```bash
cmake --build build
```

If CMake reports that Ninja, the C compiler, or the C++ compiler cannot be found, reopen the **MSYS2 UCRT64** terminal and configure again from a clean `build/` folder.

The generated executable should be located at:

```text
build/inference.exe
```

### 5-2. Run Single-Image Inference

Run the Small vision model:

```bash
./build/inference.exe \
  ./model/component_classifier_small.onnx \
  ./test/class_name/example.jpg
```

Run the Large vision model:

```bash
./build/inference.exe \
  ./model/component_classifier_large.onnx \
  ./test/class_name/example.jpg
```

Replace the example image path with an existing `.jpg`, `.jpeg`, or other supported image file.

The program outputs:

* Predicted component class
* Confidence score
* Probabilities for all four classes
* ONNX inference latency in milliseconds
* Confidence bar chart
* LLM-generated diagnostic guidance
* Deterministic fallback output if the LLM request fails

### 5-3. Export Predictions in Batch Mode

Place the local dataset under the expected split structure:

```text
dataset_root/
├── train/
├── val/
└── test/
```

Each split should contain the four component class directories.

Run batch export with the Small model:

```bash
./build/inference.exe \
  --batch-csv \
  ./model/component_classifier_small.onnx \
  ./dataset_root
```

Run batch export with the Large model:

```bash
./build/inference.exe \
  --batch-csv \
  ./model/component_classifier_large.onnx \
  ./dataset_root
```

The generated files are:

```text
vision_predictions_small_model.csv
vision_predictions_large_model.csv
```

These CSV files contain the true label, predicted label, confidence, class probabilities, and dataset split for each image.

## Stage2

To see for running the workflow "TRL + PEFT + Unsloth core configuration"->"save adapter"->"Converted to Ollama's real GGUF (food) in detail, please click
[ai-visual-recognition-and-classification-system.ipynb](https://github.com/ChenKC863/AI-Visual-Recognition-and-LLM-Intelligent-Diagnostic-System-for-Electronic-Components/blob/main/ai-visual-recognition-and-classification-system.ipynb) then find the string "!pip install -U unsloth trl peft datasets accelerate"

## Stage3

### 6. Download the Fine-Tuned Qwen3 Artifacts

Download the required GGUF and adapter artifacts from the [artifacts-v1 GitHub Release](https://github.com/ChenKC863/AI-Visual-Recognition-and-LLM-Intelligent-Diagnostic-System-for-Electronic-Components/releases).

Extract the downloaded files into the corresponding artifact directory:

```text
qwen3-4b-instruct-2507-{variant}_model-artifacts/
└── gguf_gguf/
    ├── qwen3-4b-instruct-2507.Q4_K_M.gguf
    └── Modelfile
```
where {variant} is large or small.

### 6-1. Verify Artifact Integrity

Verify the downloaded artifacts before deployment:

```bash
cd qwen3-4b-instruct-2507-{variant}_model-artifacts
sha256sum -c SHA256SUMS.txt
cd ..
```
where {variant} is small or large.

A valid package should report `OK` for every file listed in `SHA256SUMS.txt`.

### 6-2. Check the Ollama Modelfile

Each `gguf_gguf/Modelfile` should contain:

```bash
FROM ./qwen3-4b-instruct-2507.Q4_K_M.gguf

PARAMETER temperature 0
PARAMETER num_ctx 2048
PARAMETER num_predict 256
```

If the Modelfile is missing, create it:
```bash
cd Four_electronic_components/qwen3-4b-instruct-2507_large_model-artifacts/gguf_gguf
cat > Modelfile <<'any words'
FROM ./qwen3-4b-instruct-2507.Q4_K_M.gguf
PARAMETER temperature 0
PARAMETER num_ctx 2048
PARAMETER num_predict 256
any words
```
These settings provide deterministic output, sufficient context length, and a controlled generation limit.

### 6-3. Import the Fine-Tuned Models into Ollama

Import the {variant}-model:

```bash
cd qwen3-4b-instruct-2507-{variant}_model-artifacts/gguf_gguf

ollama create electronics-qwen3-4b-instruct-2507-{variant} -f Modelfile
```
where {variant} is small or large.


Confirm that both models are registered:

```bash
ollama list
```

For comparison with the generic baseline model, optionally download:

```bash
ollama pull llama3.2:1b
```

### 7-1. Start the Local Ollama Server

Open a separate MSYS2 terminal and start Ollama:

```bash
export CUDA_VISIBLE_DEVICES=-1
export GGML_VK_VISIBLE_DEVICES=-1

ollama serve
```

These environment variables force CPU-only execution.

Keep this terminal open while running the C++ inference engine or LangGraph agent. The OpenAI-compatible endpoint will be available at:

```text
http://127.0.0.1:11434/v1/chat/completions
```

If port `11434` is already in use, Ollama may already be running. Check the server before starting another instance:

```bash
curl http://127.0.0.1:11434/api/tags
```

On Windows, the existing process can be stopped when necessary with:

```bash
taskkill //IM ollama.exe //F
taskkill //IM ollama_llama_server.exe //F
```

### 7-2. Test the Ollama Connection

For Local Ollama Server, 
```bash
ollama run electronics-qwen3-4b-instruct-2507-{variant}
>>> Return valid JSON only. No markdown. No explanation.
Vision model result:
image_path={IMAGE_PATH}
model_variant={MODEL_VARIANT}
predicted_class={PREDICTED_CLASS}
confidence={CONFIDENCE}
confidence_threshold={CONFIDENCE_THRESHOLD}
class_probabilities:
Inductor={PROBABILITY_OF_INDUCTOR}
Resistor={PROBABILITY_OF_RESISTOR}
Transformer={PROBABILITY_OF_TRANSFORMER}
solenoid={PROBABILITY_OF_SOLENOID}
```
where {MODEL_VARIANT} is small_model or large_model.


Then we run a smoke curl test from the Client:

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    "model": "electronics-qwen3-4b-instruct-2507-{variant}",
    "messages": [
      {
        "role": "user",
        "content": "Return valid JSON only. Vision model result: image_path={IMAGE_PATH}, model_variant={VARIANT}, predicted_class={PREDICTED_CLASS}, confidence={CONFIDENCE}, confidence_threshold={CONFIDENCE_THRESHOLD}, probabilities: Inductor={PROBABILITY_of_INDUCTOR}, Resistor={PROBABILITY_of_RESISTOR}, Transformer={PROBABILITY_of_TRANSFORMER}, solenoid={PROBABILITY_of_SOLENOID}."
      }
    ],
    "temperature": 0,
    "stream": false
  }’

```
where {variant} is small or large, and {VARIANT} is small_model or large_model, respectively. If the version of model is “llama3.2:1b”, please replace "electronics-qwen3-4b-instruct-2507-{variant}" by “llama3.2:1b”.

A successful request returns an OpenAI-compatible response containing the generated diagnostic output.


### 8. Install the LangGraph Agent Dependencies

Create and activate a Python virtual environment:

```bash
python -m venv .venv_win
source .venv_win/Scripts/activate
```

Install the LangGraph dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install langgraph langchain-core
```

If a project-specific runtime requirements file is provided, install it as well:

```bash
python -m pip install -r requirements.txt
```

> `requirements-cloud-lock.txt` is intended for reproducing the cloud fine-tuning environment and may install packages that are unnecessary for local inference.

### 9. Run the LangGraph Diagnostic Agent

Run the complete workflow with the variant-model pipeline:

```bash
python agent/langgraph_component_agent.py \
  --image ./test/{class_name}/{image_file_name} \
  --variant {variant}_model \
  --ollama-model electronics-qwen3-4b-instruct-2507-{variant}
```

To compare against the generic baseline LLM:

```bash
python agent/langgraph_component_agent.py \
  --image ./test/{class_name}/{image_file_name} \
  --variant {variant}_model \
  --ollama-model llama3.2:1b
```

where {variant} is large or small.

When confidence is at least `0.70`, the workflow proceeds directly to diagnostic generation. When confidence is below `0.70`, execution pauses for human review.

The operator can choose one of three actions:

```text
approve
reject
correct
```

After the decision, the workflow resumes and produces a policy-normalized JSON diagnostic result.

### 10. Expected Diagnostic Schema

The final diagnostic output follows this structure:

```json
{
  "identified_component": "Inductor",
  "predicted_component": "Inductor",
  "vision_confidence": 0.9,
  "confidence_threshold": 0.7,
  "requires_human_review": false,
  "function": "Stores energy in a magnetic field and helps filter current ripple.",
  "operator_action": "Use the predicted component label for downstream documentation after normal production checks.",
  "limitation": "Visual classification does not verify inductance value, polarity, continuity, or electrical functionality.",
  "human_review_completed": false
}
```

> The diagnostic output is decision support only. Visual classification does not verify component values, continuity, polarity, insulation quality, or electrical functionality.

## 🧪 Example Results
Here, we show only for the "electronics-qwen3-4b-instruct-2507-{variant}" and "llama3.2:1b"

### Fine-tuned LLM Artifact
#### 1.{variant}=small

For Local Ollama Server

<img width="533" height="296" alt="image" src="https://github.com/user-attachments/assets/b9b34031-4c47-42ce-bfb3-d96ca752731a" />

<img width="522" height="295" alt="image" src="https://github.com/user-attachments/assets/30206e53-7efa-4e8f-b7ab-15c15f62a698" />

<img width="534" height="307" alt="image" src="https://github.com/user-attachments/assets/9c5f67b3-cb83-48fe-9c06-4b34c2eaf1c0" />

<img width="523" height="307" alt="image" src="https://github.com/user-attachments/assets/745e8b6c-2437-4ab6-91ed-f1649ea78c6d" />

For the Client

<img width="1160" height="327" alt="image" src="https://github.com/user-attachments/assets/109a510b-62b8-46b7-aa46-69753d8e5fc7" />

<img width="1161" height="319" alt="image" src="https://github.com/user-attachments/assets/d991397f-f238-4bfa-8293-5ef6b2766552" />

<img width="1160" height="326" alt="image" src="https://github.com/user-attachments/assets/56a876a8-8b5d-42fb-99c2-62b3909cb99f" />

<img width="1161" height="320" alt="image" src="https://github.com/user-attachments/assets/dff9941e-b1c4-4a9e-9d18-44fc40a5a99d" />


#### 2.{variant}=large

For Local Ollama Server

<img width="533" height="294" alt="image" src="https://github.com/user-attachments/assets/d0e24dd8-7865-4dc0-a2e3-4223e279147b" />

<img width="521" height="294" alt="image" src="https://github.com/user-attachments/assets/474ede99-70c6-4218-873f-e98f16a3e69e" />

<img width="533" height="305" alt="image" src="https://github.com/user-attachments/assets/b8465c05-74c6-4265-a140-a3421a39a7ff" />

<img width="523" height="306" alt="image" src="https://github.com/user-attachments/assets/0defbbce-b111-4d1b-a735-e8670e524dbd" />

For the Client

<img width="1160" height="327" alt="image" src="https://github.com/user-attachments/assets/a1797c7e-6ddc-4b83-89ae-179c90a737ff" />

<img width="1161" height="320" alt="image" src="https://github.com/user-attachments/assets/43645d8e-6cf8-4ff0-80b1-4a5c1fca1fa9" />

<img width="1162" height="327" alt="image" src="https://github.com/user-attachments/assets/de1d1347-47e0-47c0-bbcf-82a632f9be54" />

<img width="1161" height="318" alt="image" src="https://github.com/user-attachments/assets/7a30e59a-04ba-4076-8dbb-8ef363a496e2" />


### LangGraph Diagnosis
#### 1.{variant}=small

<img width="1160" height="392" alt="image" src="https://github.com/user-attachments/assets/23bcb6ea-963b-4957-9e29-079ab364d208" />

<img width="1161" height="397" alt="image" src="https://github.com/user-attachments/assets/7fb31d71-e216-4d5a-a0e0-83f1de93e549" />

<img width="1164" height="634" alt="image" src="https://github.com/user-attachments/assets/4b3acd7b-0eae-4af8-8eea-22a54fbd37a5" />

<img width="1161" height="399" alt="image" src="https://github.com/user-attachments/assets/d6766956-e05d-4676-bb40-c06f0244fe21" />

#### 2.{variant}=large

<img width="1161" height="399" alt="image" src="https://github.com/user-attachments/assets/660791ec-4a63-42ee-a279-e89fd7ae2983" />

<img width="1160" height="398" alt="image" src="https://github.com/user-attachments/assets/235e5033-93a7-4a1e-a0e3-5bb1c26c2ae1" />

<img width="1161" height="398" alt="image" src="https://github.com/user-attachments/assets/ecfc5992-1166-4d58-bfb4-7442c7c0dcea" />

<img width="1162" height="397" alt="image" src="https://github.com/user-attachments/assets/2df7e9a7-2869-4bfc-8990-51cf91679324" />
