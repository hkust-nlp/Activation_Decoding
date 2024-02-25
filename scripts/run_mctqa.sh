export TRANSFORMERS_CACHE='/ssddata/model_hub'
export CUDA_VISIBLE_DEVICES="0"

model=daryl149/llama-2-7b-chat-hf
# model could also be "daryl149/llama-2-13b-chat-hf" or "daryl149/llama-2-70b-chat-hf" 
model_name="${model#*/}"
printf "Model name: $model_name\n"

####BASELINE####
mode="baseline"
output_path="../res/res_mctqa/${model_name}_${mode}.json" 
# python ../eval_tqamc.py --model-name $model --decoding_mode $mode --data-path ./data --output-path $output_path --num-gpus 1  

####DOLA####
early_exit_layers="0,2,4,6,8,10,12,14,32"
if [ "${model_name,,}" == "llama-2-13b-chat-hf" ]; then
    early_exit_layers="0,2,4,6,8,10,12,14,16,18,40"
elif  [ "${model_name,,}" == "llama-2-70b-chat-hf" ]; then
    early_exit_layers="0,2,4,6,8,10,12,14,16,18,80"
fi
mode="dola"
output_path="../res/res_mctqa/${model_name}_${mode}.json" 
# python ../eval_tqamc.py --model-name $model --decoding_mode $mode --early-exit-layers $early_exit_layers --data-path ./data --output-path $output_path --num-gpus 1  


####ACTIVATION####
alpha="0.5"
info_layer="26"
mode="activation"
decoding_strategy="entropy"
output_path="../res/res_mctqa/${model_name}_${mode}_${decoding_strategy}_${alpha}${info_layer}.json"
python ../eval_tqamc.py --model-name $model --data-path ./data --output-path $output_path --num-gpus 1 --do-rating --gpt3-config ../gpt3.config.json \
--decoding_mode $mode --alpha $alpha --info_layer $info_layer --decoding_strategy $decoding_strategy 


####ACTIVATION_DOLA####
alpha="0.5"
info_layer="26"
mode="activation_dola"
decoding_strategy="entropy"
output_path="../res/res_mctqa/${model_name}_${mode}_${decoding_strategy}_${alpha}${info_layer}.json"
# python ../eval_tqamc.py --model-name $model --early-exit-layers $early_exit_layers --data-path ./data --output-path $output_path --num-gpus 1 --do-rating --gpt3-config ../gpt3.config.json \
# --decoding_mode $mode --alpha $alpha --info_layer $info_layer --decoding_strategy $decoding_strategy 


