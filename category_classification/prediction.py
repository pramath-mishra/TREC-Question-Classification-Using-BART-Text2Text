import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import re
import glob
import transformers

# Model initialization
transformers.set_seed(42)
tokenizer = transformers.BartTokenizer.from_pretrained('facebook/bart-base')
model = transformers.BartForConditionalGeneration.from_pretrained(
    max(
        glob.glob("distilgpt2-industrial-checkpoints/*"),
        key=lambda x: int(x.split("-")[-1])
    )
).cuda()


def generate_(text):
    model.eval()
    input_ids = tokenizer.encode(f"categorize: {text}", max_length=512, truncation=True, return_tensors="pt").cpu()
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    input_text = input()
    print(generate_(input_text))
