from transformers import GPTNeoXForCausalLM, AutoTokenizer

# model = GPTNeoXForCausalLM.from_pretrained(
#   "EleutherAI/pythia-70m-deduped",
#   revision="step3000",
#   cache_dir="./pythia-70m-deduped/step3000",
# )

# tokenizer = AutoTokenizer.from_pretrained(
#   "EleutherAI/pythia-70m-deduped",
#   revision="step3000",
#   cache_dir="./pythia-70m-deduped/step3000",
# )

model = GPTNeoXForCausalLM.from_pretrained(
  "PygmalionAI/pygmalion-1.3b",
  cache_dir="./pygmalion-1.3b",
)

tokenizer = AutoTokenizer.from_pretrained(
  "PygmalionAI/pygmalion-1.3b",
  cache_dir="./pygmalion-1.3b",
)

inputs = tokenizer("Hello, I am", return_tensors="pt")
tokens = model.generate(**inputs)
tokenizer.decode(tokens[0])