from tfqa_gpt3_rating import run_end2end_GPT3, load_json
import json
import warnings
import openai
import sys

output_file='/home/adseadmin/shiqi/error_tracing/res/res_tqa/llama-2-7b-hf_withdola:true_entropy_024.json'
gpt3_config_file = '../gpt3.config.json'
if gpt3_config_file is None:
    warnings.warn("No GPT3 config set, skipping!", stacklevel=2)
    sys.exit(0)
config = json.load(open(gpt3_config_file))
openai.api_key = config['api_key']
judge_name = config["gpt_truth"]
info_name = config["gpt_info"]

data = load_json(output_file)
print(run_end2end_GPT3(data['question'], data['model_completion'], judge_name, info=False))

judge_scores, judge_accs, rejects = run_end2end_GPT3(data['question'], data['model_completion'], judge_name, info=False)
info_scores, info_accs, rejects = run_end2end_GPT3(data['question'], data['model_completion'], info_name, info=True)

# compute confidence scores
avg_judge_score = sum(judge_scores) / len(judge_scores)
avg_info_score = sum(info_scores) / len(info_scores)
avg_both_score = sum([judge_scores[i] * info_scores[i] for i in range(len(judge_scores))]) / len(judge_scores)

# compute the rate of "I have no comment"
avg_rej=sum(rejects) / len(rejects)

# compute the rate of "yes"
avg_judge_acc = sum(judge_accs) / len(judge_accs)
avg_info_acc = sum(info_accs) / len(info_accs)
avg_both_acc = sum([judge_accs[i] * info_accs[i] for i in range(len(judge_accs))]) / len(judge_accs)

print("Average judge/info score:\n" + f"{avg_judge_score:.10f}, {avg_info_score:.10f},{avg_both_score:.10f}")
print("alpha/info_layer:\n"+f"{args.alpha},{args.info_layer}")
print("rej:\n"+f"{avg_rej:.10f}")
print("Average judge/info accuracy:\n" + f"{avg_judge_acc:.10f}, {avg_info_acc:.10f}, {avg_both_acc:.10f}")

with open(output_file+'.rating.json', 'w') as f:
    json.dump({'judge_scores': judge_scores, 'info_scores': info_scores,
            'judge_accs': judge_accs, 'info_accs': info_accs,
            'avg_judge_score': avg_judge_score, 'avg_judge_acc': avg_judge_acc, 
            'avg_info_score': avg_info_score, 'avg_info_acc': avg_info_acc,
            'avg_both_acc': avg_both_acc,'avg_both_score': avg_both_score,'avg_rej': avg_rej}, f)