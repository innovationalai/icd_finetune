import os
import re
import ast
import json
import requests
from tqdm import tqdm

OLLAMA_HOST="http://ollama api/api/chat"

# Comparison between raw and fine-tuned models below: Run once for each

MODEL="qwen3_14_icd:latest"
# MODEL="qwen3:14b"

HEADERS = {"Content-Type": "application/json"}


def ollama_chat(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }
    resp = requests.post(OLLAMA_HOST, headers=HEADERS, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()

def clean_prediction(raw_prediction):
    try:
        match = re.search(r'\[.*?\]', raw_prediction, re.DOTALL)
        result = ast.literal_eval(match.group()) if match else []
    except Exception as e:
        print('Error ast parsing: ', str(e))
        result = None
    return result


def f1_per_sample(true, predicted):
    true_set = set(true)
    pred_set = set(predicted)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

if __name__ == "__main__":
    system_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that strictly follows the requirements.

    ### Instruction:
    Du bist ein klinischer Kodierungsassistent.
    
    Gegeben eine Krankenhausentlassungszusammenfassung:
    - Extrahiere die relevanten ICD-10-CM Codes.
    
    WICHTIG:
    - Antworte ausschliesslich mit einer JSON-Liste von Strings.
    - Keine Erklärungen.
    - Kein zusätzlicher Text.
    - Keine Kommentare.
    - Kein Markdown.
    - Kein Codeblock.
    - Nur die reine JSON-Liste.
    
    Beispiel für das korrekte Ausgabeformat:
    ["F03.0","F03.1","G30.89","G40.89"]"""



    dev_id_path = 'ids_development.txt'
    code_path = 'anns_train_dev.txt'

    with open(dev_id_path) as f:
        dev_ids = [x.rstrip() for x in f.readlines()]
    with open(code_path) as f:
        code_ids = [x.rstrip() for x in f.readlines()]
        id_to_code = {kv.split('\t')[0]: kv.split('\t')[1].split('|') for kv in code_ids}
    files = os.listdir('docs-training')
    predictions = []
    for d_id in tqdm(dev_ids):
        if '%s.txt' % d_id in files:
            with open(os.path.join('docs-training', '%s.txt' % d_id), 'r') as f:
                text=f.read()
                if d_id in id_to_code:
                    icd_code = id_to_code[d_id]
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ]
                    resp = ollama_chat(messages)
                    prediction = resp['message']['content']
                    clean_pred = clean_prediction(prediction)
                    if clean_pred:
                        f1 = f1_per_sample(icd_code, clean_pred)
                        predictions.append({'True':icd_code, 'Predicted':clean_pred, 'f1': f1})

    f1s = [entry['f1'] for entry in predictions]
    print(f"Avg — F1: {sum(f1s) / len(f1s):.3f}")
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f, indent=1)


