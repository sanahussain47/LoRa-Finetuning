# LoRa-Finetuning
Sure! Here's the same `README.md` file **without emojis**:

---

# LoRA Fine-Tuning of FLAN-T5 for Instruction-Following Tasks

This project demonstrates how to apply **LoRA (Low-Rank Adaptation)** to fine-tune the `google/flan-t5-small` model on instruction-following tasks. While the training process used the public **Yahma/alpaca-cleaned** dataset, the evaluation was done using prompts relevant to the **Price Control and Commodities Management Department (PCCMD)** to simulate real-world domain adaptation.

---

## Project Overview

* **Base Model:** [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small)
* **Fine-Tuning Method:** [LoRA via PEFT](https://huggingface.co/docs/peft/index)
* **Instruction Dataset:** [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned) (52k instruction-following samples)
* **Domain Prompts for Evaluation:** Custom queries related to government commodity and price control

---

## How It Works

### 1. Load Base Model & LoRA Adapter

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
model = PeftModel.from_pretrained(base_model, "lora-model")
```

### 2. Merge Adapter (Optional)

```python
model = model.merge_and_unload()
```

### 3. Inference Example

```python
prompt = "What is the function of the Price Control Department?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Sample Evaluation Prompts

* What is the function of the Price Control Department?
* Why are essential commodities regulated by the government?
* What are the benefits of price monitoring for citizens?

See full outputs in `/outputs/generated_responses.csv`

---

## Key Files

| File                      | Description                                                       |
| ------------------------- | ----------------------------------------------------------------- |
| `finetune_lora.py`        | Script to apply LoRA adapter to the base model                    |
| `generate.py`             | Script to test the fine-tuned model on evaluation prompts         |
| `generated_responses.csv` | Sample outputs generated using domain-specific queries            |
| `report.pdf`              | Project summary with model details, dataset info, and reflections |

---

## Conclusion

This project shows that even with instruction data from a general domain, LoRA fine-tuning combined with prompt engineering can produce useful domain-specific outputs. It opens the door to using lightweight NLP models in public policy, citizen services, and governance workflows.

---

## Acknowledgments

* [HuggingFace PEFT](https://github.com/huggingface/peft)
* [Yahma/alpaca-cleaned Dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned)
* [Stanford Alpaca Project](https://github.com/tatsu-lab/stanford_alpaca)


