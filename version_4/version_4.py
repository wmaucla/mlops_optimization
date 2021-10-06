from transformers import DistilBertTokenizerFast, DistilBertConfig
from typing import List, Optional
import torch
from fastapi import FastAPI
import numpy as np
import transformers
from pydantic import BaseModel
import onnxruntime as ort
from torch import nn

transformers.logging.set_verbosity_error()

# Parse args
class ModelInference:
    def __init__(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('./mytokenizer/')
        self.model = ort.InferenceSession("onnx/distilbert-base-cased/model.onnx")

        config = DistilBertConfig()
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    def predict(self, message: str) -> List[np.float32]:
        inputs = self.tokenizer(message, return_tensors="np")
        onnx_output = self.model.run(["last_hidden_state"], dict(inputs))[0]
        hidden_state = torch.from_numpy(onnx_output)
        pooled_output = hidden_state[:, 0] 
        pooled_output = self.pre_classifier(pooled_output)  
        pooled_output = nn.ReLU()(pooled_output) 
        pooled_output = self.dropout(pooled_output)  
        logits = self.classifier(pooled_output) 
        return logits.detach().numpy().tolist()

class SimpleMessage(BaseModel):
    text: Optional[str] = 'test'

model_class = ModelInference()

app = FastAPI()

@app.get("/")
async def run_prediction():
    prediction = model_class.predict('This is a test message, how awesome !')
    return {'prediction': prediction}

@app.post("/prediction")
async def run_prediction(message: SimpleMessage):
    prediction = model_class.predict(message.text)
    return {'prediction': prediction}

@app.get("/health_check")
async def run_health_check():
    return {'res': True}