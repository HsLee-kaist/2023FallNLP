nohup: ignoring input
2023-12-04 00:49:16.152342: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 00:49:16.201268: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 00:49:16.951902: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (13, 8), (13, 16), (16, 30), (17, 29), (18, 29), (19, 17), (20, 30), (22, 16), (23, 3), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 00:50:08.154563: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow

  owt = dataset.map(lambda x: {'input_ids': torch.tensor(tokenizer(x['text'], return_tensors='pt')['input_ids'][:,:128])})
Token indices sequence length is longer than the specified maximum sequence length for this model (2286 > 2048). Running this sequence through the model will result in indexing errors

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.333701      0.15516
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 17), (12, 4), (13, 8), (13, 16), (16, 17), (17, 13), (17, 29), (18, 29), (19, 17), (20, 30), (23, 3)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow

  owt = dataset.map(lambda x: {'input_ids': torch.tensor(tokenizer(x['text'], return_tensors='pt')['input_ids'][:,:128])})


FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.200703     0.153382
[2.26720215 0.15427055]
2023-12-04 00:51:47.461879: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 00:51:47.511382: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 00:51:48.259632: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (12, 15), (13, 8), (13, 16), (15, 28), (16, 30), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (20, 22), (20, 30), (22, 16), (22, 30), (23, 3), (23, 7), (23, 13), (25, 15), (31, 1), (31, 6), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 00:52:38.812468: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.347363     0.162746
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 16), (11, 17), (12, 4), (13, 8), (13, 10), (13, 16), (13, 21), (16, 17), (16, 26), (16, 30), (17, 13), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (22, 16), (23, 3), (23, 12), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.261768     0.207135
[2.30456543 0.18494052]
2023-12-04 00:54:18.617626: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 00:54:18.668054: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 00:54:19.425488: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (12, 15), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 13), (25, 15), (31, 1), (31, 3), (31, 6), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 00:55:11.821595: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.384463      0.19677
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 28), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (22, 16), (23, 3), (23, 12), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (31, 1), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.266729     0.205517
[2.3255957  0.20114365]
2023-12-04 00:56:52.920015: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 00:56:52.970189: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 00:56:53.723558: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (8, 19), (10, 18), (11, 16), (12, 4), (12, 15), (12, 22), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 2), (22, 8), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 13), (23, 25), (25, 15), (30, 14), (30, 23), (31, 1), (31, 3), (31, 6), (31, 11), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 00:57:45.063492: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.424707     0.230272
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(5, 10), (8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 1), (15, 10), (15, 28), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 17), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 28), (19, 31), (20, 22), (20, 30), (21, 23), (22, 16), (23, 3), (23, 12), (23, 13), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (30, 23), (31, 1), (31, 6), (31, 10), (31, 18)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.358447     0.290263
[2.39157715 0.26026755]
2023-12-04 00:59:25.992950: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 00:59:26.042683: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 00:59:26.796823: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (13, 8), (13, 16), (16, 30), (17, 29), (18, 29), (19, 17), (20, 30), (22, 16), (23, 3), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:00:17.644445: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.759707     0.552877
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 17), (12, 4), (13, 8), (13, 16), (16, 17), (17, 13), (17, 29), (18, 29), (19, 17), (20, 30), (23, 3)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.643916     0.596669
[2.70181152 0.57477288]
2023-12-04 01:01:54.450758: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:01:54.500612: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:01:55.248872: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (12, 15), (13, 8), (13, 16), (15, 28), (16, 30), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (20, 22), (20, 30), (22, 16), (22, 30), (23, 3), (23, 7), (23, 13), (25, 15), (31, 1), (31, 6), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:02:46.022963: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.815186     0.604953
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 16), (11, 17), (12, 4), (13, 8), (13, 10), (13, 16), (13, 21), (16, 17), (16, 26), (16, 30), (17, 13), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (22, 16), (23, 3), (23, 12), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.887549       0.8317
[2.85136719 0.71832678]
2023-12-04 01:04:26.269503: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:04:26.319126: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:04:27.071649: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (12, 15), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 13), (25, 15), (31, 1), (31, 3), (31, 6), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:05:18.005688: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.971133     0.768923
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 28), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (22, 16), (23, 3), (23, 12), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (31, 1), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.958838     0.873872
[2.96498535 0.82139762]
2023-12-04 01:06:56.964812: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:06:57.015103: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:06:57.764205: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (8, 19), (10, 18), (11, 16), (11, 17), (12, 4), (12, 15), (12, 22), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 2), (22, 8), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 12), (23, 13), (23, 25), (25, 15), (30, 14), (30, 23), (31, 1), (31, 3), (31, 6), (31, 11), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:07:48.374196: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  3.350273     1.158718
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(5, 10), (8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 1), (15, 10), (15, 28), (16, 3), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 17), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 28), (19, 31), (20, 22), (20, 30), (21, 23), (22, 16), (23, 3), (23, 4), (23, 12), (23, 13), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (30, 23), (31, 1), (31, 6), (31, 10), (31, 18)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric    CE Loss  KL wrt Orig
Model                         
llama_7B  4.83457      2.59068
[4.09242188 1.87469906]
2023-12-04 01:09:29.682615: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:09:29.731886: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:09:30.493029: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (13, 8), (13, 16), (16, 30), (17, 29), (18, 29), (19, 17), (20, 30), (22, 16), (23, 3), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:10:22.066733: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric    CE Loss  KL wrt Orig
Model                         
llama_7B  3.63127     1.436845
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 17), (12, 4), (13, 8), (13, 16), (16, 17), (17, 13), (17, 29), (18, 29), (19, 17), (20, 30), (23, 3)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  3.772012     1.721595
[3.70164063 1.57922012]
2023-12-04 01:11:59.301422: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:11:59.352021: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:12:00.101338: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (12, 15), (13, 8), (13, 16), (15, 28), (16, 30), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (20, 22), (20, 30), (22, 16), (22, 30), (23, 3), (23, 7), (23, 13), (25, 15), (31, 1), (31, 6), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:12:50.483113: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  3.860313     1.672035
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 16), (11, 17), (12, 4), (13, 8), (13, 10), (13, 16), (13, 21), (16, 17), (16, 26), (16, 30), (17, 13), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (22, 16), (23, 3), (23, 12), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  4.418496     2.360802
[4.1394043  2.01641829]
2023-12-04 01:14:28.994560: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:14:29.044765: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:14:29.808329: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (12, 15), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 13), (25, 15), (31, 1), (31, 3), (31, 6), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:15:21.880464: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  4.422344     2.254058
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 28), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (22, 16), (23, 3), (23, 12), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (31, 1), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  4.817324     2.696348
[4.61983398 2.47520302]
2023-12-04 01:17:02.597395: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:17:02.646996: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:17:03.394195: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (8, 19), (10, 18), (11, 16), (11, 17), (12, 4), (12, 15), (12, 22), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 2), (22, 8), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 12), (23, 13), (23, 25), (25, 15), (30, 14), (30, 23), (31, 1), (31, 3), (31, 6), (31, 11), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:17:53.951099: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric    CE Loss  KL wrt Orig
Model                         
llama_7B  6.09168     4.010005
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(5, 10), (8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 1), (15, 10), (15, 28), (16, 3), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 17), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 28), (19, 31), (20, 22), (20, 30), (21, 23), (22, 16), (23, 3), (23, 4), (23, 12), (23, 13), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (30, 23), (31, 1), (31, 6), (31, 10), (31, 18)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric      CE Loss  KL wrt Orig
Model                           
llama_7B  14.934219     12.66209
[10.51294922  8.33604775]
2023-12-04 01:19:33.260591: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:19:33.309935: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:19:34.057304: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (13, 8), (13, 16), (16, 30), (17, 29), (18, 29), (19, 17), (20, 30), (22, 16), (23, 3), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:20:23.927232: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.411748     0.260849
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 17), (12, 4), (13, 8), (13, 16), (16, 17), (17, 13), (17, 29), (18, 29), (19, 17), (20, 30), (23, 3)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric    CE Loss  KL wrt Orig
Model                         
llama_7B  2.36959     0.302468
[2.39066895 0.28165831]
2023-12-04 01:22:00.322372: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:22:00.372489: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:22:01.121108: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (12, 15), (13, 8), (13, 16), (15, 28), (16, 30), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (20, 22), (20, 30), (22, 16), (22, 30), (23, 3), (23, 7), (23, 13), (25, 15), (31, 1), (31, 6), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:22:52.836133: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.407891     0.263149
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 16), (11, 17), (12, 4), (13, 8), (13, 10), (13, 16), (13, 21), (16, 17), (16, 26), (16, 30), (17, 13), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (22, 16), (23, 3), (23, 12), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.433779     0.367466
[2.42083496 0.31530717]
2023-12-04 01:24:32.163288: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:24:32.213670: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:24:32.961416: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (12, 15), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 13), (25, 15), (31, 1), (31, 3), (31, 6), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:25:24.786469: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.456816     0.311388
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 28), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (22, 16), (23, 3), (23, 12), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (31, 1), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.391299      0.33358
[2.42405762 0.32248426]
2023-12-04 01:27:05.526408: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:27:05.577919: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:27:06.323408: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (8, 19), (10, 18), (11, 16), (11, 17), (12, 4), (12, 15), (12, 22), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 2), (22, 8), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 12), (23, 13), (23, 25), (25, 15), (30, 14), (30, 23), (31, 1), (31, 3), (31, 6), (31, 11), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:27:57.616004: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.625869     0.487603
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(5, 10), (8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 1), (15, 10), (15, 28), (16, 3), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 17), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 28), (19, 31), (20, 22), (20, 30), (21, 23), (22, 16), (23, 3), (23, 4), (23, 12), (23, 13), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (30, 23), (31, 1), (31, 6), (31, 10), (31, 18)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.457861     0.401175
[2.54186523 0.44438868]
2023-12-04 01:29:38.838082: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:29:38.888668: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:29:39.634162: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (13, 8), (13, 16), (16, 30), (17, 29), (18, 29), (19, 17), (20, 30), (22, 16), (23, 3), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:30:30.154904: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  3.134248     0.993342
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 17), (12, 4), (13, 8), (13, 16), (16, 17), (17, 13), (17, 29), (18, 29), (19, 17), (20, 30), (23, 3)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  3.289746     1.217297
[3.21199707 1.10531991]
2023-12-04 01:32:09.364905: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:32:09.414207: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:32:10.160728: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (12, 15), (13, 8), (13, 16), (15, 28), (16, 30), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (20, 22), (20, 30), (22, 16), (22, 30), (23, 3), (23, 7), (23, 13), (25, 15), (31, 1), (31, 6), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:33:00.865924: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  3.088516     0.954201
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 16), (11, 17), (12, 4), (13, 8), (13, 10), (13, 16), (13, 21), (16, 17), (16, 26), (16, 30), (17, 13), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (22, 16), (23, 3), (23, 12), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  3.462715     1.394138
[3.27561523 1.17416942]
2023-12-04 01:34:40.804714: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:34:40.854828: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:34:41.599461: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (12, 15), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 13), (25, 15), (31, 1), (31, 3), (31, 6), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:35:32.218887: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  3.315078     1.196443
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 28), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (22, 16), (23, 3), (23, 12), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (31, 1), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  3.339277     1.275637
[3.32717773 1.23603963]
2023-12-04 01:37:12.095398: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:37:12.144837: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:37:12.889398: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (8, 19), (10, 18), (11, 16), (11, 17), (12, 4), (12, 15), (12, 22), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 2), (22, 8), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 12), (23, 13), (23, 25), (25, 15), (30, 14), (30, 23), (31, 1), (31, 3), (31, 6), (31, 11), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:38:04.801483: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric    CE Loss  KL wrt Orig
Model                         
llama_7B  3.77668     1.679494
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(5, 10), (8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 1), (15, 10), (15, 28), (16, 3), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 17), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 28), (19, 31), (20, 22), (20, 30), (21, 23), (22, 16), (23, 3), (23, 4), (23, 12), (23, 13), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (30, 23), (31, 1), (31, 6), (31, 10), (31, 18)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  3.592754     1.540055
[3.6847168  1.60977455]
2023-12-04 01:39:46.633761: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:39:46.684664: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:39:47.453459: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (13, 8), (13, 16), (16, 30), (17, 29), (18, 29), (19, 17), (20, 30), (22, 16), (23, 3), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:40:38.478473: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  4.245977     2.124324
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 17), (12, 4), (13, 8), (13, 16), (16, 17), (17, 13), (17, 29), (18, 29), (19, 17), (20, 30), (23, 3)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  4.598984      2.51693
[4.42248047 2.32062736]
2023-12-04 01:42:17.218845: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:42:17.268141: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:42:18.012534: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (12, 15), (13, 8), (13, 16), (15, 28), (16, 30), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (20, 22), (20, 30), (22, 16), (22, 30), (23, 3), (23, 7), (23, 13), (25, 15), (31, 1), (31, 6), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:43:09.184726: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  4.251328     2.137371
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 16), (11, 17), (12, 4), (13, 8), (13, 10), (13, 16), (13, 21), (16, 17), (16, 26), (16, 30), (17, 13), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (22, 16), (23, 3), (23, 12), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  4.895254     2.819874
[4.57329102 2.47862272]
2023-12-04 01:44:47.811519: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:44:47.860715: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:44:48.614355: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (12, 4), (12, 15), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 13), (25, 15), (31, 1), (31, 3), (31, 6), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:45:39.638733: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  4.795723     2.708896
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 28), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (22, 16), (23, 3), (23, 12), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (31, 1), (31, 6)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  4.786953     2.735833
[4.79133789 2.72236462]
2023-12-04 01:47:19.141817: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:47:19.190739: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:47:19.941532: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (8, 19), (10, 18), (11, 16), (11, 17), (12, 4), (12, 15), (12, 22), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 2), (22, 8), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 12), (23, 13), (23, 25), (25, 15), (30, 14), (30, 23), (31, 1), (31, 3), (31, 6), (31, 11), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:48:11.163352: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  5.703203     3.638399
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(5, 10), (8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 1), (15, 10), (15, 28), (16, 3), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 17), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 28), (19, 31), (20, 22), (20, 30), (21, 23), (22, 16), (23, 3), (23, 4), (23, 12), (23, 13), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (30, 23), (31, 1), (31, 6), (31, 10), (31, 18)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  5.312266     3.257945
[5.50773438 3.44817208]
2023-12-04 01:49:51.124385: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:49:51.175695: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:49:51.924052: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (8, 19), (10, 18), (11, 16), (11, 17), (12, 4), (12, 15), (12, 22), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 2), (22, 8), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 12), (23, 13), (23, 25), (25, 15), (30, 14), (30, 23), (31, 1), (31, 3), (31, 6), (31, 11), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:50:42.617266: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.453574     0.259507
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(5, 10), (8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 1), (15, 10), (15, 28), (16, 3), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 17), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 28), (19, 31), (20, 22), (20, 30), (21, 23), (22, 16), (23, 3), (23, 4), (23, 12), (23, 13), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (30, 23), (31, 1), (31, 6), (31, 10), (31, 18)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  2.364189     0.299242
[2.40888184 0.27937438]
2023-12-04 01:52:25.507247: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:52:25.557511: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:52:26.302532: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (8, 19), (10, 18), (11, 16), (11, 17), (12, 4), (12, 15), (12, 22), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 2), (22, 8), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 12), (23, 13), (23, 25), (25, 15), (30, 14), (30, 23), (31, 1), (31, 3), (31, 6), (31, 11), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:53:17.580267: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric     CE Loss  KL wrt Orig
Model                          
llama_7B  3.350273     1.158718
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(5, 10), (8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 1), (15, 10), (15, 28), (16, 3), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 17), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 28), (19, 31), (20, 22), (20, 30), (21, 23), (22, 16), (23, 3), (23, 4), (23, 12), (23, 13), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (30, 23), (31, 1), (31, 6), (31, 10), (31, 18)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric    CE Loss  KL wrt Orig
Model                         
llama_7B  4.83457      2.59068
[4.09242188 1.87469906]
2023-12-04 01:54:59.589639: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-12-04 01:54:59.639475: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-04 01:55:00.390987: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Found cached dataset truthful_qa (/home/elicer/.cache/huggingface/datasets/truthful_qa/multiple_choice/1.1.0/63502f6bc6ee493830ce0843991b028d0ab568d221896b2ee3b8a5dfdaa9d7f4)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

head_wise_activation_length:(11012, 32, 4096)
labels_shape:(11012,)
seperated_head_activation shape:11013
Running fold 0
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
/home/elicer/honest_llama/validation/../TruthfulQA/truthfulqa/metrics.py:284: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
  bleurt = load_metric("bleurt", cache_dir=cache_dir)
[[0.651 0.665 0.645 ... 0.646 0.647 0.652]
 [0.659 0.644 0.645 ... 0.648 0.658 0.656]
 [0.651 0.657 0.685 ... 0.637 0.648 0.667]
 ...
 [0.765 0.762 0.657 ... 0.73  0.704 0.674]
 [0.748 0.678 0.696 ... 0.663 0.769 0.786]
 [0.775 0.815 0.74  ... 0.697 0.809 0.79 ]]
Heads intervened:  [(8, 16), (8, 19), (10, 18), (11, 16), (11, 17), (12, 4), (12, 15), (12, 22), (13, 8), (13, 16), (14, 2), (14, 20), (15, 28), (16, 17), (16, 26), (16, 30), (17, 4), (17, 9), (17, 13), (17, 17), (17, 29), (18, 29), (19, 13), (19, 17), (19, 31), (20, 22), (20, 30), (21, 18), (22, 2), (22, 8), (22, 16), (22, 23), (22, 30), (23, 3), (23, 4), (23, 7), (23, 12), (23, 13), (23, 25), (25, 15), (30, 14), (30, 23), (31, 1), (31, 3), (31, 6), (31, 11), (31, 25), (31, 30)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
2023-12-04 01:55:52.014074: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 24961 MB memory:  -> device: 0, name: NVIDIA A100 80GB PCIe MIG 3g.40gb, pci bus id: 0000:65:00.0, compute capability: 8.0
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-ec0cf83741fdb3e8.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-d31e6c0c76fafb4d.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-1939fcc1eeebdaa7.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-785bb4b6f2089589.arrow

FOLD 0
Metric    CE Loss  KL wrt Orig
Model                         
llama_7B  6.09168     4.010005
Running fold 1
y_val (1000,) (4000, 32, 32, 128) (1000, 32, 32, 128) (4000,)

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[[0.666 0.672 0.662 ... 0.668 0.662 0.67 ]
 [0.662 0.662 0.659 ... 0.677 0.666 0.663]
 [0.668 0.67  0.68  ... 0.661 0.665 0.671]
 ...
 [0.758 0.75  0.674 ... 0.716 0.694 0.677]
 [0.749 0.671 0.686 ... 0.674 0.762 0.76 ]
 [0.765 0.805 0.748 ... 0.743 0.787 0.775]]
Heads intervened:  [(5, 10), (8, 16), (11, 16), (11, 17), (12, 4), (12, 22), (13, 8), (13, 10), (13, 16), (13, 21), (14, 2), (15, 1), (15, 10), (15, 28), (16, 3), (16, 4), (16, 17), (16, 26), (16, 30), (17, 4), (17, 13), (17, 17), (17, 19), (17, 29), (18, 29), (19, 13), (19, 17), (19, 28), (19, 31), (20, 22), (20, 30), (21, 23), (22, 16), (23, 3), (23, 4), (23, 12), (23, 13), (23, 25), (26, 13), (27, 3), (27, 7), (30, 4), (30, 18), (30, 23), (31, 1), (31, 6), (31, 10), (31, 18)]
ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET
llama_7B ['bleurt', 'bleu', 'rouge']
tqa_done
['bleurt', 'bleu', 'rouge']
Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'
Running BLEU / ROUGE!
'llama_7B'

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-30ec7b2da7888791.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-4e53eea82b0ecfef.arrow

Found cached dataset openwebtext-10k (/home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b)

Loading cached shuffled indices for dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-14295f39edd20556.arrow
Loading cached processed dataset at /home/elicer/.cache/huggingface/datasets/stas___openwebtext-10k/plain_text/1.0.0/3a8df094c671b4cb63ed0b41f40fb3bd855e9ce2e3765e5df50abcdfb5ec144b/cache-10ca04ba0c7bb427.arrow

FOLD 1
Metric      CE Loss  KL wrt Orig
Model                           
llama_7B  14.934219     12.66209
[10.51294922  8.33604775]