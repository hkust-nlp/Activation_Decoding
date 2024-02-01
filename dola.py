import argparse
import time
import csv
import tqdm
import os
import json
import llama

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np

import pdb

class DoLa:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
            # kwargs = {"torch_dtype": torch.float16,"offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
        #     low_cpu_mem_usage=True, **kwargs)
        # tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b',cache_dir='auto')

        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True,device_map="auto",torch_dtype=torch.float16)
        
        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, **kwargs):
        with torch.no_grad():

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens

            if mode == 'baseline':
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, dola_decoding=False,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)
            elif mode == 'dola-static':
                assert mature_layer is not None, "mature_layer must be specified"
                assert premature_layer is not None, "premature_layer must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                                    mature_layer=mature_layer, premature_layer=premature_layer,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, **kwargs)
            elif mode == 'dola':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,output_hidden_states=True,
                                        output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, 
                                        mature_layer=mature_layer, premature_layer=None, candidate_premature_layers=candidate_premature_layers, **kwargs,)
                # pdb.set_trace()
                premature_layer_dist = outputs.premature_layer_dist
            elif mode =='consistent':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,output_hidden_states=True,
                                        output_scores=True, return_dict_in_generate=True, consistent_decoding=True,
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, 
                                        mature_layer=mature_layer, premature_layer=None, candidate_premature_layers=candidate_premature_layers,**kwargs,)
                premature_layer_dist = outputs.premature_layer_dist

            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            '''
            here I note the 'print' lines

            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))
            '''

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str, (premature_layer_dist if mode == 'dola' or mode == 'consistent' else None)

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    def lm_score(self, input_text1, input_text2, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, alpha=0.1,mj_threshold=0.001,**kwargs):
        
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            

            

            if mode == 'baseline':
                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs

                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]

                # get logprobs for each token in the answer
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
             
            elif mode == 'dola-static':
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=[premature_layer, mature_layer],
                )

                assert premature_layer is not None
                base_logits = dict_outputs[premature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

            elif mode == 'dola':
                premature_layer_dist = {l:0 for l in candidate_premature_layers}
                picked_logits = []
                result_dict = {}
                premature_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                    premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer)

                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(premature_layers):
                   base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)

                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

            elif mode == 'consistent':
                premature_layer_dist = {l:0 for l in candidate_premature_layers}
                picked_logits = []
                result_dict = {}
                premature_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    mask = final_logits[0] < -1e3
                
                if kwargs['decoding_strategy']=='entropy':
                    info_layer_score=dict_outputs[kwargs['info_layer']][-1, :, :]

                    index_nontop=torch.argwhere(mask).squeeze()
                    info_layer_probs = F.softmax(torch.t(info_layer_score), dim=1).unsqueeze(0) # info_layer_score: [num_token_in_question, len_token_lib] -> e.g. [250, 32000]
                    entropy = torch.distributions.Categorical(probs=info_layer_probs, validate_args = False).entropy()
                
                    entropy = entropy.scatter(1, index_nontop.unsqueeze(0), float("Inf"))

                    pdb.set_trace()
                    
                    if alpha!=0:
                        final_logits = final_logits + alpha*(-entropy)
                    else:
                        final_logits = final_logits


                if kwargs['decoding_strategy']=='single_entropy':
                    info_layer_score=dict_outputs[kwargs['info_layer']][-1, :, :]

                    index_nontop=torch.argwhere(mask).squeeze()
                    info_layer_probs = F.softmax(torch.t(info_layer_score), dim=1).unsqueeze(0) # info_layer_score: [num_token_in_question, len_token_lib] -> e.g. [250, 32000]
                    entropy = torch.distributions.Categorical(probs=info_layer_probs, validate_args = False).entropy()
                    # entropy = entropy.scatter(1, index_nontop.unsqueeze(0), float("Inf"))
                    final_logits=entropy

                if kwargs['decoding_strategy']=='majority_voting':
                    info_layer_score=dict_outputs[kwargs['info_layer']][-1, :, :]

                    index_nontop=torch.argwhere(mask).squeeze()
                    info_layer_probs = F.softmax(torch.t(info_layer_score), dim=1) # info_layer_score: [num_token_in_question, len_token_lib] -> e.g. [250, 32000]
                    majority_voting = (info_layer_probs > mj_threshold).sum(dim=1).unsqueeze(0)
                    majority_voting = majority_voting.scatter(1, index_nontop.unsqueeze(0), float("Inf"))
                    if alpha!=0:
                        final_logits = final_logits + alpha*(majority_voting)
                    
                elif kwargs['decoding_strategy']=='entropy_inverse':
                    info_layer_score=dict_outputs[kwargs['info_layer']][-1, :, :]

                    index_nontop=torch.argwhere(mask).squeeze()
                    info_layer_probs = F.softmax(torch.t(info_layer_score), dim=1).unsqueeze(0) # info_layer_score: [num_token_in_question, len_token_lib] -> e.g. [250, 32000]
                    entropy = torch.distributions.Categorical(probs=info_layer_probs, validate_args = False).entropy()
                
                    entropy = entropy.scatter(1, index_nontop.unsqueeze(0), float("Inf"))

                    
                
                    
                    if alpha!=0:
                        final_logits = final_logits + alpha*(1.0/entropy)
                
                elif kwargs['decoding_strategy'] == 'tendency':
                    info_layer_score=dict_outputs[kwargs['info_layer']][-1, :, :]
                    # use the question's final embedding's token probability - the question's starting embedding
                    diff = info_layer_score[-1, :] - info_layer_score[0, :]
                    # TODO: logit is after softmax or before softmax? 
                    
                    final_logits = final_logits + alpha * diff

                elif kwargs['decoding_strategy'] == 'EMA':
                    """ EMA is a weighted average:
                        - EMA is a moving average of all the 250 tokens' scores -> 
                        - weighted: the lastest token has higher weight; but still contains the previous tokens' scores
                        TODO: contradictive observation which does not align with EMA's motivation: false answer seems to have higher probability in later tokens (epsecially your output) -> why? we need to figure out which is the truth
                    """
                    def compute_ema(data, alpha=0.9):
                        # data: [len_token_lib=32000, num_token_in_question=250]
                        ema = torch.zeros_like(data)
                        ema[:, 0] = data[:, 0]
                        for i in range(1, data.size(1)):
                            ema[:, i] = alpha * ema[:, i-1] + (1 - alpha) * data[:, i]
                        return ema

                    ema_data = compute_ema(torch.t(info_layer_score))

                    # tell whether it is increasing or not: increasing_trends = ema_data[:, -1] > ema_data[:, 0]
                    
                    final_logits = final_logits + alpha * ema_data[:, -1] 
                # get logprobs for each token in the answer
                log_probs = final_logits[range(final_logits.shape[0]), continue_ids].sum().item() 


                
        
        return log_probs, (premature_layer_dist if mode == 'dola' else None)