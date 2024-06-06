import torch
from run_acc import *

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    million_params = total_params / 1_000_000
    return million_params

model1=RobertaDUMASpanClassifier.from_pretrained(
    "../../MODEL/roberta"
)
model2=RobertaForQuestionAnswering.from_pretrained(
    "../../MODEL/roberta"
)
params1=count_parameters(model1)
params2=count_parameters(model2)

print(f"cls: {params1}")
print(f"cor: {params2}")