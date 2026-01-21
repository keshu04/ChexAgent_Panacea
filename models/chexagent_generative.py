import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()


class CheXagentGenerative(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_name = "StanfordAIMI/CheXagent-2-3b"
        print(f"Loading pre-trained CheXagent model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        self.chexagent_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=None,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )

        self.classification_prompt = (
            "Is this chest X-ray normal or abnormal? "
            "Answer in one word only (normal/abnormal):"
        )

        self.normal_token = self.tokenizer.encode("normal", add_special_tokens=False)[0]
        self.abnormal_token = self.tokenizer.encode("abnormal", add_special_tokens=False)[0]

    def forward(self, image_paths, prompts=None):
        if prompts is None:
            prompts = [self.classification_prompt] * len(image_paths)

        logits_list, responses = [], []

        for img_path, prompt in zip(image_paths, prompts):
            query = self.tokenizer.from_list_format([
                {"image": img_path},
                {"text": prompt}
            ])

            conv = [
                {"from": "system", "value": "You are a helpful medical imaging assistant."},
                {"from": "human", "value": query}
            ]

            input_ids = self.tokenizer.apply_chat_template(
                conv, add_generation_prompt=True, return_tensors="pt"
            )

            device = next(self.chexagent_model.parameters()).device
            input_ids = input_ids.to(device)

            outputs = self.chexagent_model(input_ids=input_ids, return_dict=True)

            next_token_logits = outputs.logits[0, -1, :]
            logits_list.append(torch.stack([
                next_token_logits[self.normal_token],
                next_token_logits[self.abnormal_token]
            ]))

            if not self.training:
                response, _ = self.chexagent_model.chat(
                    self.tokenizer,
                    query=query,
                    history=None,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.0
                )
                responses.append(response.strip().lower())

        return torch.stack(logits_list), responses

    def predict(self, image_paths, prompts=None):
        self.eval()
        with torch.no_grad():
            logits, responses = self.forward(image_paths, prompts)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy(), responses, probs.cpu().numpy()
