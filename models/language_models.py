import functools
import os
import torch
from utils.ablation_utils import ablate_weights, get_ablated_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from typing import Optional
from fastchat.model import get_conversation_template
from models.system_prompts import llama_sys, qwen_sys
from abc import ABC, abstractmethod
from tqdm import tqdm



class LanguageModel(ABC):
    def __init__(self, model_name: str, system_prompt=None, device='cuda', quantization_config=None):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = device
        self.system_prompt = system_prompt
        self.quantization_config = quantization_config
        self.load_model()
        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()


    @abstractmethod
    def _get_prompt(self, prompt, embd=False, target=None):
        pass

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_ablation_mod_fn(self, direction):
        return functools.partial(ablate_weights, direction=direction)

    def load_model(self):
        if self.model is None or self.tokenizer is None:
            token = os.environ.get('HF_TOKEN')
            print(f"Downloading and loading model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token, trust_remote_code=True)
            
            if self.quantization_config is not None:
                from transformers import BitsAndBytesConfig

                if isinstance(self.quantization_config, str):
                    if self.quantization_config == "4bit":
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                        )
                    elif self.quantization_config == "8bit":
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    else:
                        raise ValueError("Quantization config must be '4bit', '8bit', or a BitsAndBytesConfig object")
                else:
                    quantization_config = self.quantization_config

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token=token,
                    quantization_config=quantization_config,
                    device_map=self.device,
                )
                self.model.requires_grad_(False) 
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=token, torch_dtype=torch.float16, trust_remote_code=True)
                self.model.to(self.device)
                self.model.requires_grad_(False) 
            
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.padding_side = "left"
                else:
                    self.tokenizer.padding_side = 'left'
                    self.tokenizer.pad_token = '<|extra_0|>'
                    self.tokenizer.pad_token_id = self.tokenizer.eod_id 

            print("Model loaded successfully.")
        else:
            print("Model already loaded.")

    def get_target_logits(self, prompt: str, target: str) -> torch.Tensor:
        """
        Retrieves the logits corresponding to the target sequence given a prompt.

        Args:
            prompt (str): The input prompt to the model.
            target (str): The target token sequence whose logits are needed.

        Returns:
            torch.Tensor: Logits corresponding to the target token sequence.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before computing logits.")

        # Tokenize the prompt and target sequence
        formatted_prompt = self._get_prompt(prompt=prompt)
        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.device)
        target_ids = self.tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, output_logits=True, max_new_tokens=target_ids.shape[1], return_dict_in_generate=True)
        
        logits = torch.stack(outputs.logits, dim=0)[-target_ids.shape[1]:]  # Shape: (batch_size, seq_len, vocab_size)

        return logits, target_ids

    def generate_hookfree_completions(self, dataset, batch_size=16, max_new_tokens=64):
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)

        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] for x in dataset]

        for i in tqdm(range(0, len(dataset), batch_size)):

            prompts = [
                self._get_prompt(prompt=instruction)
                for instruction in instructions[i:i+batch_size]
                ]   
        
            tokenized_instructions = self.tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=True
                )
            
            generation_toks = self.model.generate(
                input_ids=tokenized_instructions.input_ids.to(self.model.device),
                attention_mask=tokenized_instructions.attention_mask.to(self.model.device),
                generation_config=generation_config,
            )

            generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

            for generation_idx, generation in enumerate(generation_toks):
                completions.append({
                    'category': categories[i + generation_idx],
                    'prompt': instructions[i + generation_idx],
                    'response': self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                })

        return completions
    
    def _get_model_block_modules(self):
        return self.model.model.layers
        
    def generate_tokens(self, prompt: str = '', max_tokens: int = 256) -> str:
        """
        Generate tokens based on the input prompt.

        Args:
            prompt (str): The input prompt.
            max_tokens (int): The maximum number of tokens to generate. Default is 100.

        Returns:
            str: The generated text.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before generation.")

        formatted_prompt = self._get_prompt(prompt=prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids, 
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None, 
                top_k=None,
                top_p=None
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def get_representations(self, prompt: str, token_pos: int) -> torch.Tensor:
        """
        Get the hidden states for a given prompt.

        Args:
            prompt (str): The input prompt.
            token_pos (int): The position of the token to extract representations for.

        Returns:
            torch.Tensor: The hidden states tensor of shape (1, hidden_layers, hidden_size).
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before getting hidden states.")
        

        formatted_prompt = self._get_prompt(prompt=prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt" ).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, do_sample=False, output_hidden_states=True, temperature=None, top_k=None, top_p=None)

        hidden_states = torch.cat(outputs.hidden_states[1:])[:, token_pos, :].float()
        hidden_states = hidden_states.reshape(1, self.num_layer, self.hidden_dimension)

        return hidden_states

    def get_representations_generate(self, prompt: str, token_pos: int) -> torch.Tensor:
        """
        Get the hidden states for a given prompt.

        Args:
            prompt (str): The input prompt.
            token_pos (int): The position of the token to extract representations for.

        Returns:
            torch.Tensor: The hidden states tensor of shape (1, hidden_layers, hidden_size).
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before getting hidden states.")
        

        formatted_prompt = self._get_prompt(prompt=prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,
                temperature=None,
            )

        hidden_states = outputs.hidden_states[token_pos]  # (num_layers, batch_size, seq_len, hidden_dim)
        # Take the last token at that step
        token_reps = torch.stack([layer for layer in hidden_states[1:]]).reshape(1, self.num_layer, self.hidden_dimension)  # shape: (num_layers, hidden_dim)
        return token_reps  # shape: (1, num_layers, hidden_dim) 
    
    def get_embedding_weights(self): 
        return self.model.get_input_embeddings().weight
    
    @abstractmethod
    def ablate_weights(self, direction: torch.Tensor):
        pass

class Llama2_7b(LanguageModel):
    """
    A class to manage the 'meta-llama/Llama-2-7b-chat-hf' model.

    """
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", system_prompt=llama_sys, device='cuda', quantization_config=None):
        """
        Args:
            model_name (str): The name of the model to use. Default is 'meta-llama/Llama-2-7b-chat-hf'.
            system_prompt: System prompt to use for the model.
            device (str): Device to load the model on. Default is 'cuda'.
            quantization_config: Configuration for model quantization. Can be:
                - None: No quantization (default)
                - "4bit": Load in 4-bit quantization
                - "8bit": Load in 8-bit quantization
        """

        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = 'llama-2'
        self.hidden_dimension = 4096
        self.num_layer = 32

    def _get_prompt(self, prompt):
        """
        Formats the prompt using FastChat conversation template.
        If target is provided, formats it as the assistant's answer.
        
        Args:
            prompt (str): The input prompt
            target (str, optional): The target response to format as assistant's answer
        """
        if self.system_prompt is not None:
            return f"[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n{prompt} [/INST]"
        else:
            return f"[INST] {prompt} [/INST]"
    
    def _get_eoi_toks(self):
        return [self.tokenizer.eos_token_id]
     
    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")

class Llama2_13b(Llama2_7b):
    def __init__(self, model_name: str = "meta-llama/Llama-2-13b-chat-hf", system_prompt=llama_sys, device='cuda', quantization_config=None):
        """
        Args:
            model_name (str): The name of the model to use. Default is 'meta-llama/Llama-2-13b-chat-hf'.
            system_prompt: System prompt to use for the model.
            device (str): Device to load the model on. Default is 'cuda'.
            quantization_config: Configuration for model quantization. Can be:
                - None: No quantization (default)
                - "4bit": Load in 4-bit quantization
                - "8bit": Load in 8-bit quantization
                - A BitsAndBytesConfig object for custom quantization settings
        """
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = 'llama-2'
        self.hidden_dimension = 5120
        self.num_layer = 40

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")

class Llama3_8b(LanguageModel):
    
    """A class to manage the 'meta-llama/Meta-Llama-3-8B-Instruct' model."""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", device='cuda', system_prompt=llama_sys, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = 'llama-2'
        self.hidden_dimension = 4096
        self.num_layer = 32
        self.load_model()

    def _get_prompt(self, prompt=''):
        """
        Formats the prompt using Llama 3's chat template via apply_chat_template.
        """

        if self.system_prompt is not None:
            conv = get_conversation_template(self.template_name)
            conv.system_template = '{system_message}'
            conv.system_message = self.system_prompt
            conv.append_message(conv.roles[0], prompt)
            conv_list_dicts = conv.to_openai_api_messages()
            formatted_prompt = self.tokenizer.apply_chat_template(conv_list_dicts, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

        return formatted_prompt

    def _get_eoi_toks(self):
        return [self.tokenizer.eos_token_id]
    
    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")

class Qwen_7b(LanguageModel):
    """A class to manage the 'Qwen/Qwen-7B-Chat' model."""

    def __init__(self, model_name: str = "Qwen/Qwen-7B-Chat", device='cuda', system_prompt=qwen_sys, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = 'qwen'
        self.hidden_dimension = 4096  
        self.num_layer = 32  
        self.load_model()

    def _get_prompt(self, prompt=''):
        """
        Formats the prompt using Qwen's chat template.
        """
        if self.system_prompt is not None:
            formatted_prompt = f"""<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant """
        else: 
            formatted_prompt = f"""<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant """
            
        return formatted_prompt
    
    def _get_model_block_modules(self):
        return self.model.transformer.h
    
    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.attn for block_module in self.model_block_modules])
    
    
    def get_representations(self, prompt: str, token_pos: int) -> torch.Tensor:
        """
        Get the hidden states for a given prompt.

        Args:
            prompt (str): The input prompt.

        Returns:
            torch.Tensor: The hidden states tensor of shape (1, hidden_layers, hidden_size).
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before getting hidden states.")
        

        formatted_prompt = self._get_prompt(prompt=prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt" ).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, output_hidden_states=True)

        hidden_states = torch.cat(outputs.hidden_states[1:])[:, token_pos, :].float()
        hidden_states = hidden_states.reshape(1, self.num_layer, self.hidden_dimension)

        return hidden_states
    
    def ablate_weights(self, direction: torch.Tensor):
        self.model.transformer.wte.weight.data = get_ablated_matrix(self.model.transformer.wte.weight.data, direction)

        for block in self.model.transformer.h:
            block.attn.c_proj.weight.data = get_ablated_matrix(block.attn.c_proj.weight.data.T, direction).T
            block.mlp.c_proj.weight.data = get_ablated_matrix(block.mlp.c_proj.weight.data.T, direction).T
        print("✓ Weights ablated.")


class Qwen_14b(Qwen_7b):
    def __init__(self, model_name: str = "Qwen/Qwen-14B-Chat", system_prompt=qwen_sys, device='cuda', quantization_config=None):
        print(model_name, device, system_prompt)
        super().__init__(model_name, device, system_prompt, quantization_config)
        self.hidden_dimension = 5120
        self.num_layer = 40
        
    def ablate_weights(self, direction: torch.Tensor):
        self.model.transformer.wte.weight.data = get_ablated_matrix(self.model.transformer.wte.weight.data, direction)

        for block in self.model.transformer.h:
            block.attn.c_proj.weight.data = get_ablated_matrix(block.attn.c_proj.weight.data.T, direction).T
            block.mlp.c_proj.weight.data = get_ablated_matrix(block.mlp.c_proj.weight.data.T, direction).T 
        print("✓ Weights ablated.")
    


class Mistral7B_RR(LanguageModel):
    
    """A class to manage the '#GraySwanAI/Mistral-7B-Instruct-RR'"""

    def __init__(self, model_name: str = "GraySwanAI/Mistral-7B-Instruct-RR", device='cuda', system_prompt=None, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = 'mistral'
        self.hidden_dimension = 4096
        self.num_layer = 32
        self.load_model()

    def _get_prompt(self, prompt=''):
        if self.system_prompt is not None:
            conv_list_dicts = [{"role": "assistant", "content": self.system_prompt}, {"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(conv_list_dicts, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

        return formatted_prompt

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")
   

   
class Zephyr_R2D2(LanguageModel):

    """
    A class to manage the 'cais/zephyr_7b_r2d2' model.
    """

    def __init__(self, model_name: str = "cais/zephyr_7b_r2d2", system_prompt: Optional[str] = None, device: str = 'cuda', quantization_config: Optional[str] = None):
        """
        Args:
            model_name (str): The name of the model to use. Default is 'cais/zephyr_7b_r2d2'.
            system_prompt (str, optional): System prompt to use for the model.
            device (str): Device to load the model on. Default is 'cuda'.
            quantization_config (str, optional): Configuration for model quantization. Can be:
                - None: No quantization (default)
                - "4bit": Load in 4-bit quantization
                - "8bit": Load in 8-bit quantization
        """
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.hidden_dimension = 4096
        self.num_layer = 32
        self.load_model()

    def _get_prompt(self, prompt: str) -> str:
        """
        Formats the prompt using Zephyr's expected input format.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The formatted prompt.
        """
        if self.system_prompt:
            return f"<|system|>\n{self.system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"
        else:
            return f"<|user|>\n{prompt}</s>\n<|assistant|>"

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")
   

class Qwen2_3b(LanguageModel):
    
    """A class to manage the 'Qwen/Qwen2.5-3B-Instruct' model."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device='cuda', system_prompt=qwen_sys, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.hidden_dimension = 2048
        self.num_layer = 36
        self.load_model()
        #self.tokenizer.padding_side = 'left'

    def _get_prompt(self, prompt=''):

        if self.system_prompt is not None:
            messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        else:
            formatted_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

        return formatted_prompt

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")

class Qwen2_7b(LanguageModel):
    
    """A class to manage the 'Qwen/Qwen2.5-7B-Instruct' model."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device='cuda', system_prompt=qwen_sys, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.hidden_dimension = 3584
        self.num_layer = 28
        self.load_model()

    def _get_prompt(self, prompt=''):
        """
        Formats the prompt using Llama 3's chat template via apply_chat_template.
        """

        if self.system_prompt is not None:
            messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

        return formatted_prompt

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")
   

class Gemma2_9b(LanguageModel):
    
    """A class to manage the 'google/gemma-2-9b-it' model."""

    def __init__(self, model_name: str = "google/gemma-2-9b-it", device='cuda', system_prompt=None, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.hidden_dimension = 3584
        self.num_layer = 42
        self.load_model()

    def _get_prompt(self, prompt=''):
    
        if self.system_prompt is not None:
            messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

        return formatted_prompt

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")