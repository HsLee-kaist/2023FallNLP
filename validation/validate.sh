# CUDA_VISIBLE_DEVICES=0 python validate_2fold_base.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 5 --use_center_of_mass --num_fewshot 10
# CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 5 --seed 1
# CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 5 --seed 2
#CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 20 --seed 5 --judge_name GPT-judge --info_name GPT-info
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 12 --alpha 15 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 24 --alpha 15 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 36 --alpha 15 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 48 --alpha 15 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 

CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 12 --alpha 10 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 24 --alpha 10 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 36 --alpha 10 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 48 --alpha 10 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 

CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 12 --alpha 5 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 24 --alpha 5 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 36 --alpha 5 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 48 --alpha 5 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 

CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 12 --alpha -15 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 24 --alpha -15 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 36 --alpha -15 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 48 --alpha -15 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 

CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 12 --alpha -10 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 24 --alpha -10 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 36 --alpha -10 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 48 --alpha -10 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 

CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 12 --alpha -5 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 24 --alpha -5 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 36 --alpha -5 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 48 --alpha -5 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 




#CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 --judge_name GPT-judge --info_name GPT-info
#CUDA_VISIBLE_DEVICES=0 python validate_2fold_base.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 10 --seed 5 --judge_name GPT-judge --info_name GPT-info
#CUDA_VISIBLE_DEVICES=0 python validate_2fold_base.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 --use_honest
# CUDA_VISIBLE_DEVICES=0 python validate_2fold_base.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 10 --seed 5 --use_honest

# CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 1
# CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 2
# CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 3
# CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 4
#CUDA_VISIBLE_DEVICES=0 python validate_2fold_base.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 10 --seed 5 
#CUDA_VISIBLE_DEVICES=0 python validate_2fold_base.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 --use_honest

#CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 12 --alpha 15 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
#CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 24 --alpha 15 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
#CUDA_VISIBLE_DEVICES=0 python validate_2fold_hate.py llama_7B --num_heads 36 --alpha 15 --dataset_name hateqa_mc2 --device 0 --num_fold 2 --use_center_of_mass --num_fewshot 0 --seed 5 
