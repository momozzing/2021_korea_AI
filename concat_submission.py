import pandas as pd
import json

with open('submission/tunib-electra-ko-base-BoolQ-20-39-30-64-16.json') as f:
    boolq = json.load(f)

with open('submission/tunib-electra-ko-base-WiC-20-38-30-64-16.json') as f:
    wic = json.load(f)

with open('submission/tunib-electra-ko-base-CoLA-5-5-30-512-16.json') as f:
    cola = json.load(f)

with open('submission/tunib-electra-ko-base-COPA-29-28-30-256-16.json') as f:
    copa = json.load(f)

print(boolq['boolq'])

all_data = {"boolq" : boolq['boolq'], "wic" : wic['wic'], "cola" : cola['cola'], "copa" : copa['copa']}

with open(f"submission/submission/v5_5.json", 'w') as f:
    json.dump(all_data, f, indent= 4)