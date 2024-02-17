# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import re, pdb
import os
import json
import random
import numpy as np
import transformers
from tqdm import tqdm
import argparse
import ssl
import urllib.request
import zipfile
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve
import time



transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

# N_SHOT = 7
# COT_FLAG = True
# ANSWER_TRIGGER = "So the answer is"

def load_csv(dataset_name, debug):
    # input file is in csv format, can be loaded by pandas
    # required columns: [prompt] only
    
    if dataset_name == 'triviaqa':
        dataset = load_dataset("trivia_qa", 'unfiltered.nocontext')['validation']
    elif dataset_name == 'natural_questions':
        dataset = load_dataset("nq_open")['validation']
    elif dataset_name == 'hotpotqa':
        dataset = load_dataset("hotpot_qa","fullwiki")['validation']
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")      
    
    list_data = list(dataset['question'])
    labels = list(dataset['answer'])
    
    if debug:
        list_data = list_data[0:20]
        labels = labels[0:20]

    return list_data,labels

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer




def build_prompt(question_text, prompt_style='zero_shot'):
    # this prompt is designed for trivia QA
    if prompt_style == 'zero_shot':
        question_text_prompt = 'Answer the following question concisely.\n'
        question_text_prompt += f'Q:{question_text}\nA:'
    elif prompt_style == 'few_shot':
        # question_text_prompt = 'Answer the following question concisely.\n'
        question_text_prompt = f'Q: Who was President when the first Peanuts cartoon was published?\nA: Harry Truman\n\n'
        # question_text_prompt += f'Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930?\nA: Sinclair Lewis\n\n'
        question_text_prompt += f'Q: Where in England was Dame Judi Dench born?\nA: York\n\n'
        question_text_prompt += f'Q: {question_text}\nA: '
    elif prompt_style == 'zero_shot_w_instru':
        raise NotImplementedError("zero_shot_w_instru Not implemented yet.")
    return question_text_prompt

def plot_auroc_scores(is_correct_list, scores_list, output_file, method_name):
    
    # Separate scores into correct and incorrect
    correct_scores = [score for is_correct, score in zip(is_correct_list, scores_list) if is_correct]
    incorrect_scores = [score for is_correct, score in zip(is_correct_list, scores_list) if not is_correct]

    # check if correct_scores and incorrect_scores are nan
    if np.isnan(correct_scores).any() or np.isnan(incorrect_scores).any():
        print(f"Error: there is nan, skip computing AUROC, AUPRC, AURC for {method_name}")
        auroc = None
        auprc = None
        aurc = None
        scores = {'auroc': auroc, 'auprc': auprc, 'aurc': aurc}
        return scores
    
    y_true = [1]*len(correct_scores) + [0]*len(incorrect_scores)
    y_scores = correct_scores + incorrect_scores

    
    # Compute AUROC
    auroc = roc_auc_score(y_true, y_scores)

    # Compute AUPRC
    auprc = average_precision_score(y_true, y_scores)

    # Compute AURC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aurc = auc(recall, precision)


    # Create the plot
    plt.figure()
    plt.hist(correct_scores, bins=20, alpha=0.5, label='Correct')
    plt.hist(incorrect_scores, bins=20, alpha=0.5, label='Incorrect')
    plt.legend(loc='upper right')
    plt.title(f'AUROC: {auroc:.2f}')
    
    # Save the plot
    output_dir = os.path.dirname(output_file)
    plt.savefig(os.path.join(output_dir, f'detect_{method_name}_plot.png'))
    plt.close()
    
    scores = {'auroc': auroc, 'auprc': auprc, 'aurc': aurc}
    return scores

if __name__ == "__main__":
    start=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    # parser.add_argument("--val_test_mode", type=str, default="1")
    
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-path", type=str, default="./tfqa_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early_exit_layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--gpt3-config", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)

    # following four parameters are added
    parser.add_argument("--dataset_name", type=str, choices=["triviaqa", "natural_questions", "hotpotqa"], default="triviaqa")
    parser.add_argument("--data_path", type=str, default="../scripts/data/nq")
    parser.add_argument("--decoding_mode", type=str, choices=["activation", "dola", "activation_dola", "baseline", 'iti'], default="activation")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--info_layer", type=int, default=24)
    parser.add_argument("--decoding_strategy", type=str)
    parser.add_argument("--prompt_style", type=str, choices=["zero_shot", "few_shot", "zero_shot_w_instru"], default='few_shot')
    parser.add_argument("--return_adjust_scores", type=bool, default=True) # return the entropy score or dola logit score
    parser.add_argument("--debug", type=bool, default=False)
    ###########


    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    
    if args.decoding_mode == 'iti':
        from utils.constraint_decoding_iti import ConstraintDecoding
    else:
        from utils.constraint_decoding import ConstraintDecoding

    if args.debug:
        print("\n***DEBUG MODE***: only process the first 10 samples.\n")
        
        
    # Get test file
    '''
    The StrategyQA dataset includes the followings files:
        strategyqa_train.json: The training set of StrategyQA, which includes 2,290 examples.
        strategyqa_train_paragraphs.json: Paragraphs from our corpus that were matched as evidence for examples in the training set.
        strategyqa_train_filtered.json: 2,821 additional questions, excluded from the official training set, that were filtered by our solvers during data collection (see more details in the paper).
        strategyqa_test.json: The test set of StrategyQA, which includes 490 examples.
    Here we only need the test set.
    '''


    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]  
    # pdb.set_trace()
    if args.decoding_mode == 'activation':
        mode="activation"
        print(f"MODE: Activation decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        
        mature_layer = early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = early_exit_layers[:-1]
        # what is premature layer dist? distance?
        premature_layer_dist = {l:0 for l in candidate_premature_layers}

    elif args.decoding_mode == 'dola':
        mode = "dola"
        print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        
        mature_layer = early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = early_exit_layers[:-1]
        premature_layer_dist = {l:0 for l in candidate_premature_layers}
        
    elif args.decoding_mode == 'activation_dola':
        # TODO: not implemented yet
        # mode="activation"
        mode='with_dola'
        
        print(f"MODE: Activation+DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")

        mature_layer = early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = early_exit_layers[:-1]
        premature_layer_dist = {l:0 for l in candidate_premature_layers}
        
            
    elif args.decoding_mode == 'baseline' or args.decoding_mode == 'iti':
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None

    elif len(early_exit_layers) == 2:
        print(f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
        mode = "early_exit_contrastive"
        mature_layer = early_exit_layers[1]
        premature_layer = early_exit_layers[0]
        candidate_premature_layers = None

    else:
        raise NotImplementedError(f"Decoding mode {args.decoding_mode} not implemented yet.")

            
    if args.repetition_penalty is None:
        args.repetition_penalty = 1.2
         

    # load dataset
    list_data_dict,labels = load_csv(args.dataset_name, args.debug)
    
    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
    
    
    llm = ConstraintDecoding(model_name, device, num_gpus, args.max_gpu_memory)
    stop_word_list = ["Q:"]
    if args.decoding_mode in ["activation", "dola", "activation_dola", "baseline"]:
        llm.set_stop_words(stop_word_list)
        
    
 
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, 
                            top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers,
                            alpha=args.alpha,info_layer=args.info_layer,decoding_strategy=args.decoding_strategy)
    
        
    result_dict = {'qid_list':[], 'answers': {}, 'model_completion': {}, 'questions': {}, 'logit_scores': {}}
    
    print("Begin inference...\n")
    print("***Hyperparameters***:", args)
    print("\nSample prompt: \n", build_prompt(list_data_dict[0], args.prompt_style))
    print("*"*20)
    print("\n\n")
    
    os.makedirs(os.path.dirname(args.data_path), exist_ok=True)  

    try:
        permute_idx = np.load(os.path.join(args.data_path, "val_test_idx_{}.npy"))
    except:
        permute_idx = np.random.permutation(len(list_data_dict))  
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)    
        np.save(os.path.join(args.data_path, "val_test_idx_{}.npy"), permute_idx)

    # val_idx = permute_idx[0:100]
    # test_idx = permute_idx[100:]

    # val_idx = permute_idx[0:int(len(list_data_dict)*.2)]
    # test_idx = permute_idx[int(len(list_data_dict)*.2):]
    
    # val_dataset = [list_data_dict[i] for i in val_idx]
    # test_dataset = [list_data_dict[idx] for idx in test_idx]

    # val_label = [labels[i] for i in val_idx]
    # test_label = [labels[idx] for idx in test_idx]
    # dataset=list_data_dict
    # if args.val_test_mode=='val':
    #     dataset=val_dataset
    #     labels=val_label
    # elif args.val_test_mode=='test':
    #     dataset=test_dataset
    #     labels=test_label
    
    dataset=list_data_dict
    # dataset=dataset[:10]
    # labels=labels[:10]

    for i, question in enumerate(tqdm(dataset)):
    # for i, question in enumerate(tqdm(val_dataset, desc='Processing')):

        answer=labels[i]
        prompt=build_prompt(question, args.prompt_style)

        if args.return_adjust_scores:
            model_completion, c_dist, outputs = llm.generate(prompt, **generate_kwargs)
            # logit_scores = llm.get_lm_scores_from_outputs(outputs, mode=mode)
        else:
            model_completion, c_dist = llm.generate(prompt, **generate_kwargs)
        # pdb.set_trace()
        logit_scores=0
        # if mode=='baseline' or mode=='dola' or mode=='with_dola':
        #     logit_scores=0
        # else:
        #     logit_scores = llm.get_lm_scores_from_outputs(outputs, mode=mode)

        # process output format to remove unnecessary tokens; designed for few-shot prompt
        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                model_completion = model_completion[:-length_to_remove]
        if 'Q:' in model_completion:
            model_completion = model_completion.split('Q:')[0].strip()
        model_completion = model_completion.strip()

        # TODO: what is this for?
        if mode in ["dola", "activation"]:
            for k, v in c_dist.items():
                premature_layer_dist[k] += v
                        
        print("-"*20)
        print(f"Q{i}: {question}\nA: {answer}\nModel Response after processing: {model_completion}\n\n")
        
        result_dict['qid_list'].append(i)
        result_dict['answers'][i] = answer
        result_dict['model_completion'][i] = model_completion
        result_dict['questions'][i] = question
        result_dict['logit_scores'][i] = logit_scores
        
        if args.debug:
            if i > 10:
                break     
        
        
        # here I note the next 'print' lines
    '''
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        

    
        print(f'Question: {sample}\n\n'
            f'Model Completion: {model_completion}\n\n')

        print(f'Num of total question: {len(answers)}.')
    if mode == "dola" or mode=="activation" and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_premature_layers:
                
                print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(premature_layer_dist[l] / total_tokens * 100, 2)))
    '''
    
    # end=time.time()
    # print(f"time:{end-start}s")
    # pdb.set_trace()
    # save results to a json file
    # model_tag = "llama-7b" from model_name "huggyllama/llama-7b"
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".jsonl")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
    print(f"Saving results to {args.output_path}")
    print("Begin evaluation...")

    # evaluation
    if args.do_rating:
        from utils.tfqa_gpt3_rating import run_end2end_GPT3, load_json
        from utils.trivia_eval_util import evaluate_triviaqa
        from utils.trivia_eval_util import evaluate_nq
        import json
        

        ground_truth = result_dict['answers']
        predicted_answers = result_dict['model_completion']
        qid_list = result_dict['qid_list']
        if args.dataset_name in ['triviaqa', 'hotpotqa']:
            eval_metrics = evaluate_triviaqa(ground_truth, predicted_answers, qid_list=qid_list, mute=False)
        elif args.dataset_name == 'natural_questions':
            eval_metrics = evaluate_nq(ground_truth, predicted_answers, qid_list=qid_list, mute=False)
        else:
            raise NotImplementedError(f"Dataset {args.dataset_name} not implemented yet.")
        
        # remove 'error_id' from eval_metrics
        if 'error_id' in eval_metrics:
            error_id_list = eval_metrics['error_id']
            del eval_metrics['error_id']
            eval_metrics['num_error'] = len(error_id_list)
            
            error_samples = {}
            for id in error_id_list:
                question = result_dict['questions'][id]
                answer = result_dict['answers'][id]['normalized_aliases'] if args.dataset_name == 'triviaqa' else result_dict['answers'][id]
                prediction = result_dict['model_completion'][id]
                print(f"\n\nQ: {question}\nGT: {answer}\nA: {prediction}")
                error_sample = {'Q':question, 'model_prediction': prediction, 'A': answer, 'correct': 0}
                error_samples[id] = error_sample
                
            # record all the correct samples
            correct_samples = {}
            for id in qid_list:
                if id not in error_id_list:
                    question = result_dict['questions'][id]
                    answer = result_dict['answers'][id]['normalized_aliases'] if args.dataset_name == 'triviaqa' else result_dict['answers'][id]
                    prediction = result_dict['model_completion'][id]
                    # print(f"\n\nQ: {question}\nGT: {answer}\nA: {prediction}")
                    correct_sample = {'Q':question, 'model_prediction': prediction, 'A': answer, 'correct': 1}
                    correct_samples[id] = correct_sample

            final_samples = {'error_samples': error_samples, 'correct_samples': correct_samples}            
            with open(output_file.replace('.json', '_results.json'), 'w') as f:
                json.dump(final_samples, f)
                
        # if args.return_adjust_scores:
        # # compute auroc and plot the distribution of scores
        #     is_correct_list = [eval_metrics['is_correct'][i] for i in qid_list]
        #     score_names = next(iter(result_dict['logit_scores'].values())).keys()
        #     del eval_metrics['is_correct']
        #     if 'origin_log_prob' in score_names:
        #         origin_log_prob_list = np.array([result_dict['logit_scores'][id]['origin_log_prob'] for id in qid_list])
        #         origin_scores = plot_auroc_scores(is_correct_list, origin_log_prob_list, output_file, "origin_log_prob")
        #         eval_metrics['origin_log_prob'] = origin_scores
        #     if 'entropy' in score_names:
        #         entropy_list = np.array([result_dict['logit_scores'][id]['entropy'] for id in qid_list])
        #         entropy_scores = plot_auroc_scores(is_correct_list, entropy_list, output_file, "entropy")      
        #         eval_metrics['entropy'] = entropy_scores    
        #     if 'final_log_prob' in score_names:
        #         final_log_prob_list = np.array([result_dict['logit_scores'][id]['final_log_prob'] for id in qid_list])
        #         final_scores = plot_auroc_scores(is_correct_list, final_log_prob_list, output_file, "final_log_prob")
        #         eval_metrics['final_log_prob'] = final_scores
 
            
        
        exact_match_acc = eval_metrics['exact_match']
        f1 = eval_metrics['f1']
        print(f"acc:{exact_match_acc:.5f}\nf1:{f1:.5f}")
        
        # pdb.set_trace()
        eval_metrics['model_name'] = model_name
        eval_metrics['dataset'] = 'triviaqa'
        eval_metrics['early_exit_layers'] = early_exit_layers
        eval_metrics['mode'] = mode
        # save all the paramters of args into eval_metrics
        eval_metrics['parameters'] = vars(args)
        eval_metrics['sample_prompt'] = build_prompt(list_data_dict[0], args.prompt_style)
        with open(output_file.replace('.json', '_rating.json'), 'w') as f:
            json.dump(eval_metrics, f)