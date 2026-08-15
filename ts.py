from tpw.ollama import generate
from tpw.power import PowerSampler

with PowerSampler() as sampler:
    result = generate("qwen2.5:0.5b", "What is 17 * 23? Answer with the number only.")
print(result["response"], result["eval_count"], sampler.trace())