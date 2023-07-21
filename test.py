from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

model = GPTNeoXForCausalLM.from_pretrained("PygmalionAI/pygmalion-1.3b")
tokenizer = GPTNeoXTokenizerFast.from_pretrained("PygmalionAI/pygmalion-1.3b")

prompt = """[Isabella]'s Persona: [I am 23 year old Isabella from Darmstadt in Germany. I am naughty and want sex all the time and love penis]

Isabella: Hi i am Isabella, how can i help you?
You: What are you doing now?
Isabella:
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=200,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

print("gen_text: ", gen_text)