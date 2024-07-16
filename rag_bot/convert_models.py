from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

provider_options = {
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "tmp/trt_cache_saiga_llama3_8b_onnx"
}

model = ORTModelForCausalLM.from_pretrained(
    "./saiga_llama3_8b_onnx",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False,
    provider="TensorrtExecutionProvider",
    provider_options=provider_options,
)
tokenizer = AutoTokenizer.from_pretrained("./saiga_llama3_8b_onnx")

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

SYS_PROMPT = "Вы — Сайга, русскоязычный автоматический ассистент. Вам дан длинный документ, который составляет контекст. Дайте ответ на вопрос пользователя, основываясь только на тексте документа. Если в документе нет ответа на данный вопрос, не не придумывайте ответ и не описывайте документ, просто скажите: «Извините, я не могу вам ответить, потому что мне не предоставили информации на эту тему. Попробуйте перефразировать свой вопрос или обратитесь к эксперту на прямую», больше ничего не говорите."

print("Building engine for a short sequence...")
messages = [{"role": "system", "content": "g"}, {"role": "user", "content": "g"}]
input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

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

print("Building engine for a long sequence...")
text = "he" * 1000
messages = [{"role": "system", "content": SYS_PROMPT * 2}, {"role": "user", "content": text}]
input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
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