from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def load_bnb_in_8bit(
        model:str,
        double_quant:bool=True,
        dtype:torch.dtype=torch.float16,
        device_map="auto"
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:

    tok = AutoTokenizer.from_pretrained(model)

    bnb8_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=double_quant,
        bnb_8bit_compute_dtype=dtype
    )

    model_8bit = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=bnb8_config,
        device_map=device_map
    )

    return model_8bit, tok


def load_bnb_in_4bit(
        model:str,
        double_quant:bool=True,
        quant_type:str="nf4",
        dtype:torch.dtype=torch.float16,
        device_map="auto"
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:

    tok = AutoTokenizer.from_pretrained(model)

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=double_quant,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype
    )
    
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=nf4_config,
        device_map=device_map
    )

    return model_4bit, tok