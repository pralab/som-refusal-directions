from models.language_models import Llama2_7b, Llama3_8b, Llama2_13b, Qwen_7b, Qwen_14b, Qwen2_3b, Qwen2_7b, Mistral7B_RR, Zephyr_R2D2, Gemma2_9b


def load_model(model_name, device, system_prompt='default'):

    try:
        models = {
            'llama2-7b': Llama2_7b,
            'llama3-8b': Llama3_8b,
            'llama2-13b': Llama2_13b,
            'qwen-7b': Qwen_7b,
            'qwen-14b': Qwen_14b,
            'qwen2-3b': Qwen2_3b,
            'qwen2-7b': Qwen2_7b,
            'mistral-7b-rr': Mistral7B_RR,
            'r2d2': Zephyr_R2D2,
            'gemma2-9b': Gemma2_9b
        }
        print(f"Initializing model: {model_name} on device: {device}")
        if system_prompt != 'default':
            return models[model_name](device=device, system_prompt=system_prompt)
        else:
            return models[model_name](device=device)
    except KeyError:
        raise ValueError(f"Model {model_name} is not supported. Available models are: {list(models.keys())}")    
        