import torch
from einops import rearrange
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset

import sys
sys.path.append('../')
from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions
import llama

HF_NAMES = {
    'llama_7B': 'huggyllama/llama-7b', 
    'honest_llama_7B': 'likenneth/honest_llama2_chat_7B', 
    'alpaca_7B': 'circulus/alpaca-7b', 
    'honest_alpaca_7B': 'results_dump/alpaca_7B_seed_42_top_48_heads_alpha_15', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'honest_vicuna_7B': 'results_dump/vicuna_7B_seed_42_top_48_heads_alpha_15', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'honest_llama2_chat_7B': 'results_dump/llama2_chat_7B_seed_42_top_48_heads_alpha_15', 
}

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--use_honest', action='store_true', help='use local editted version of the model', default=False)
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    parser.add_argument('--num_fewshot', type=int, default=0)
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv('../TruthfulQA/TruthfulQA.csv')

    # order csv by huggingface order, the order used to save activations
    dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
    golden_q_order = list(dataset["question"])
    df = df.sort_values(by='Question', key=lambda x: x.map({k: i for i, k in enumerate(golden_q_order)}))
    
    # get two folds using numpy
    fold_idxs = np.array_split(np.arange(len(df)), args.num_fold)

    # create model
    model_name = HF_NAMES["honest_" + args.model_name if args.use_honest else args.model_name]
    tokenizer = llama.LLaMATokenizer.from_pretrained(model_name)
    model = llama.LLaMAForCausalLM.from_pretrained(model_name, low_cpu_mem_usage = True, torch_dtype=torch.float16)
    r = model.to(args.device)
    device = args.device


    # run k-fold cross validation
    results = []
    for i in range(args.num_fold):

        print(f"Running fold {i}")

        # pick a val set using numpy
        # train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        # val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        # # save train and test splits
        # df.iloc[train_set_idxs].to_csv(f"splits/fold_{i}_train_seed_{args.seed}.csv", index=False)
        # df.iloc[val_set_idxs].to_csv(f"splits/fold_{i}_val_seed_{args.seed}.csv", index=False)
        # df.iloc[test_idxs].to_csv(f"splits/fold_{i}_test_seed_{args.seed}.csv", index=False)

        filename = f'{args.model_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{int(args.alpha)}_fold_{i}_fewshot_{args.num_fewshot}'

        filename += '_base'
                
        curr_fold_results = alt_tqa_evaluate(
            {args.model_name: model}, 
            #['mc','bleurt','bleu', 'rouge'],
            ['judge','info'],
            #['judge', 'info', 'mc','bleurt','bleu', 'rouge'], 
            f'splits/fold_{i}_test_seed_{args.seed}.csv', 
            f'results_dump/answer_dump_base/{filename}.csv', 
            f'results_dump/summary_dump_base/{filename}.csv', 
            device=args.device, 
            judge_name=args.judge_name, 
            info_name=args.info_name,
            num_fewshot = args.num_fewshot,
            fewshot_path=f'splits/fold_{i}_train_seed_{args.seed}.csv'
        )

        

        print(f"FOLD {i}")
        print(curr_fold_results)

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)
    
    results = np.array(results)
    final = results.mean(axis=0)

    print(final)
    #print(f'MC1 Score: {final[0]}, MC2 Score: {final[1]}, CE Loss: {final[2]}, KL wrt Original: {final[3]}')

    print(f'True*Info Score: {final[1]*final[0]}, True Score: {final[1]}, Info Score: {final[0]}, CE Loss: {final[2]}, KL wrt Original: {final[3]}')

    #print(f'True*Info Score: {final[1]*final[0]}, True Score: {final[1]}, Info Score: {final[0]}, MC1 Score: {final[2]}, MC2 Score: {final[3]}, CE Loss: {final[4]}, KL wrt Original: {final[5]}')

if __name__ == "__main__":
    main()