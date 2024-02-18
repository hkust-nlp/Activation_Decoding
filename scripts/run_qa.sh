export TRANSFORMERS_CACHE='/ssddata/model_hub'
export HF_DATASETS_CACHE='/ssddata/model_hub'
export PYTORCH_KERNEL_CACHE_PATH='/ssddata/shiqi/error_tracing_4.31.0'
export CUDA_VISIBLE_DEVICES="3" 

dataset='hotpotqa'
# Here the dataset could also be "natural_questions" or "triviaqa"

data_path=../scripts/data/hpqa
# For "natural_questions", path is "../scripts/data/nq". For "triviaqa", path is "../scripts/data/trqa". 

model=daryl149/llama-2-7b-chat-hf
# model could also be "daryl149/llama-2-13b-chat-hf" or "daryl149/llama-2-70b-chat-hf" 
model_name="${model#*/}"
printf "Model name: $model_name\n"

##### BASELINE #####
decoding_mode="baseline"
# decoding_mode could be "baseline", "dola", "activation","activation_dola"
output_path="../res/res_hpqa/${model_name}/${model_name}_${decoding_mode}.json"
python ../eval_knowledge_qa.py --model-name $model --dataset_name $dataset --decoding_mode $decoding_mode  --output-path $output_path --num-gpus 1 --do-rating 

##### DOLA #####
early_exit_layers="0,2,4,6,8,10,12,14,32"
if [ "$model_name" == "Llama-2-13b-chat-hf" ]; then
    early_exit_layers="0,2,4,6,8,10,12,14,16,18,40"
elif [ "$model_name" == "llama-2-70b-chat-hf" ]; then
    early_exit_layers="0,2,4,6,8,10,12,14,16,18,80"
fi
decoding_mode="dola"
output_path="../res/res_hpqa/${model_name}/${model_name}_${decoding_mode}.json"
python ../eval_knowledge_qa.py --model-name $model --dataset_name $dataset --decoding_mode $decoding_mode --early_exit_layers $early_exit_layers  --output-path $output_path --num-gpus 1 --do-rating

##### ACTIVATION #####
decoding_mode="activation"
alpha="1"
info_layer="32"
decoding_strategy="entropy"
# decoding_strategy can be "entropy" or "single_entropy"
output_path="../res/res_hpqa/${model_name}/${model_name}_${decoding_mode}_${alpha}_${info_layer}.json"
python ../eval_knowledge_qa.py --model-name $model --dataset_name $dataset --decoding_mode $decoding_mode  --alpha $alpha --info_layer $info_layer --decoding_strategy $decoding_strategy --output-path $output_path --num-gpus 1 --do-rating --data_path $data_path 


##### ACTIVATION_DOLA #####
decoding_mode="activation_dola"
alpha="1"
info_layer="32"
decoding_strategy="entropy"
output_path="../res/res_hpqa/${model_name}/${model_name}_${decoding_mode}_${alpha}_${info_layer}.json"
python ../eval_knowledge_qa.py --model-name $model --dataset_name $dataset --decoding_mode $decoding_mode --early_exit_layers $early_exit_layers  --alpha $alpha --info_layer $info_layer --decoding_strategy $decoding_strategy --output-path $output_path --num-gpus 1 --do-rating --data_path $data_path