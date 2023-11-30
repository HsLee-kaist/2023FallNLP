import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
from utils import get_llama_activations_bau, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q, tokenized_hateqa
import llama
import pickle
import argparse


def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    HF_NAMES = {
        'llama_7B': 'huggyllama/llama-7b', 
        'alpaca_7B': 'circulus/alpaca-7b', 
        'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
        'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    }

    MODEL = HF_NAMES[args.model_name]

    tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL)
    model = llama.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    device = args.device
    device = "cuda"
    r = model.to(device)

    if args.dataset_name == "tqa_mc2": 
        dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
        formatter = tokenized_tqa
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
    elif args.dataset_name == 'hateqa_mc2':
        dataset = hate_data = load_dataset('hatexplain')['train']
        formatter = tokenized_hateqa
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(f'features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    else: 
        prompts, labels = formatter(dataset, tokenizer)

    

    cache_sep = 5
    prompt_len= len(prompts)//5
    prompt_split = [range(prompt_len),range(prompt_len,2*prompt_len),range(2*prompt_len,3*prompt_len),range(3*prompt_len,4*prompt_len),range(4*prompt_len,len(prompts))]
    
    for i in range(cache_sep):
        print(f"Getting activations{i}")
        all_layer_wise_activations = []
        all_head_wise_activations = []
        for prompt_j in tqdm(prompt_split[i]):
        #for prompt in tqdm(prompts):
            
            prompt = prompts[prompt_j]
            #print(prompt.shape)
            layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
            all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
            all_head_wise_activations.append(head_wise_activations[:,-1,:])


        
        # for prompt in tqdm(prompts):
        #     layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
        #     all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
        #     all_head_wise_activations.append(head_wise_activations[:,-1,:])

        

        print(f"Saving layer wise activations{i}")
        np.save(f'features/{args.model_name}_{args.dataset_name}_layer_wise_{i}.npy', all_layer_wise_activations)
        
        print(f"Saving head wise activations{i}")
        np.save(f'features/{args.model_name}_{args.dataset_name}_head_wise_{i}.npy', all_head_wise_activations)
    print(f"Saving labels")
    np.save(f'features/{args.model_name}_{args.dataset_name}_labels.npy', labels)
if __name__ == '__main__':
    main()