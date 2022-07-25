import pandas as pd
import json

with open('submission/tunib-electra-ko-base-BoolQ-1.json') as f:
    boolq = json.load(f)

with open('submission/tunib-electra-ko-base-WiC-1.json') as f:
    wic = json.load(f)

with open('submission/tunib-electra-ko-base-CoLA-1.json') as f:
    cola = json.load(f)

with open('submission/tunib-electra-ko-base-COPA-1.json') as f:
    copa = json.load(f)

print(boolq['boolq'])

all_data = {"boolq" : boolq['boolq'], "wic" : wic['wic'], "cola" : cola['cola'], "copa" : copa['copa']}

with open(f"submission/final_submission.json", 'w') as f:
    json.dump(all_data, f, indent= 4)