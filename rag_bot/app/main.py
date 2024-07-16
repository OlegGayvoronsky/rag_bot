import gradio as gr
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

bi_encoder = SentenceTransformer('cointegrated/rubert-tiny2', device=device)
bi_encoder.max_seq_length = 512
cross_encoder = CrossEncoder('DiTy/cross-encoder-russian-msmarco', device=device, max_length=512)
top_k = 10
dataset = load_dataset("Str4nnik/wikipedia_with_embedings", revision="small_3000_embedded")
corpus_embeddings = torch.tensor(dataset['train']['embeddings'])
model_id = "IlyaGusev/saiga_llama3_8b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

SYS_PROMPT = "Вы — Сайга, русскоязычный автоматический ассистент. Вам дан длинный документ, который составляет контекст. Дайте ответ на вопрос пользователя, основываясь только на тексте документа. Если в документе нет ответа на данный вопрос, не не придумывайте ответ и не описывайте документ, просто скажите: «Извините, я не могу вам ответить, потому что мне не предоставили информации на эту тему. Попробуйте перефразировать свой вопрос или обратитесь к эксперту на прямую», больше ничего не говорите."

def search(query, k=1):
    # Semantic Search #
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)[0]

    # Re-Ranking #
    cross_inp = [[query, dataset['train']['text'][hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    scores, documents = [], []
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in hits[0:k]:
        scores.append(hit['cross-score'])
        documents.append(dataset['train']['text'][hit['corpus_id']].replace("\n", " "))
    return scores, documents


def format_prompt(prompt, documents, k=1):
    prompt = f"Вопрос:{prompt}\nКонтекст:"
    for idx in range(k):
        prompt += f"{documents[idx]}\n"
    return prompt


def generate(formatted_prompt):
    formatted_prompt = formatted_prompt[:2000]
    messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": formatted_prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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

    response = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(response, skip_special_tokens=True).replace('Вот разговорный ответ на ваш вопрос:\n\n', '')


def rag_bot(message, history):
    scores, documents = search(message)
    formatted_prompt = format_prompt(message, documents)
    return generate(formatted_prompt)


gr.ChatInterface(
    rag_bot,
    chatbot=gr.Chatbot(placeholder="<strong>Ваш персональный умный помощник</strong><br>Если у вас есть вопросы, "
                                   "я могу помочь найти на них ответы"),
    textbox=gr.Textbox(placeholder="Задайте свой вопрос", container=False, scale=7),
    title="Умный помощник",
    examples=["Расскажи мне про Юрия Гагарина", "Расскажи, кто такой Белобог?", "Расскажи побольше про игру Ящер"],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Удалить предыдущее сообщение",
    clear_btn="Очистить",
).launch(share=True)
