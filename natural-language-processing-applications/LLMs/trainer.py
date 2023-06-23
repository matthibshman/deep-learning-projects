import torch
import torch.nn.functional as F
from transformers import Trainer
import numpy as np
from helpers import INCORRECT, CORRECT, FORGOTTEN


class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        if not hasattr(self, "classified_status"):
            self.classified_status = {}

        model.eval()
        predictions = torch.argmax(F.softmax(model(**inputs)[1]), dim=1)
        labels = inputs["labels"]
        correct = inputs["input_ids"][predictions == labels]
        incorrect = inputs["input_ids"][predictions != labels]
        self.classify_status(correct, self.handle_correct)
        self.classify_status(incorrect, self.handle_incorrect)

        model.train()
        return super().training_step(model, inputs)

    def classify_status(self, examples, handle):
        for i in range(0, examples.size()[0]):
            sentence = np.array2string(examples[i].numpy(), max_line_width=10000)
            handle(sentence)

    def handle_correct(self, sentence):
        if sentence not in self.classified_status or self.classified_status[sentence] == INCORRECT:
            self.classified_status[sentence] = CORRECT

    def handle_incorrect(self, sentence):
        if sentence not in self.classified_status:
            self.classified_status[sentence] = INCORRECT
        elif self.classified_status[sentence] == CORRECT:
            self.classified_status[sentence] = FORGOTTEN
