from optimum.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from optimum.onnxruntime import ORTModelForSequenceClassification
import torch

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
# )
SYS_PROMPT = "Вы — Сайга, русскоязычный автоматический ассистент. Вам дан длинный документ, который составляет контекст. Дайте ответ на вопрос пользователя, основываясь только на тексте документа. Если в документе нет ответа на данный вопрос, не не придумывайте ответ и не описывайте документ, просто скажите: «Извините, я не могу вам ответить, потому что мне не предоставили информации на эту тему. Попробуйте перефразировать свой вопрос или обратитесь к эксперту на прямую», больше ничего не говорите."
model = AutoModelForCausalLM.from_pretrained(
  "IlyaGusev/saiga_llama3_8b",
   torch_dtype=torch.bfloat16,
   # export=True,
   # provider="CUDAExecutionProvider",
   device_map='auto'
   # quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/saiga_llama3_8b")

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Building engine for a short sequence...")
messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": "Кто такой Марко Ройс?"}]
input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
print("1")
attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(model.device)
pad_token_id = tokenizer.eos_token_id
outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        eos_token_id=terminators,
        pad_token_id=pad_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
print("2")
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))