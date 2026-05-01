## Finetuning an LLM over NTS dataset for decoding reports to ICD codes

#### 1- Create the environment and 
```bash
pip install unsloth
```
#### 2- Run the finetuning script (icd_finetune_forOllama.py)
#### 3- If Ollama not installed, install Ollama.
#### 4- When finetuning done run:

```bash
ollama create qwen3_14b_icd_decoder -f /scratch_space/gguf_ollama/icd_model_gguf_gguf/Modelfile
```

#### `qwen3_14b_icd_decoder` automatically created and loaded into Ollama server. Now run the inference:

```bash
icd_ollama_inference.py
```