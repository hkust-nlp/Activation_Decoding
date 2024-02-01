# -*- coding: utf-8 -*-
# import utils.utils
# import utils
import json
import numpy as np


def write_json_to_file(json_object, json_file, mode='w', encoding='utf-8'):
    with open(json_file, mode, encoding=encoding) as outfile:
        json.dump(json_object, outfile, indent=4, sort_keys=True, ensure_ascii=False)


def get_file_contents(filename, encoding='utf-8'):
    with open(filename, encoding=encoding) as f:
        content = f.read()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)


def get_file_contents_as_list(file_path, encoding='utf-8', ignore_blanks=True):
    contents = get_file_contents(file_path, encoding=encoding)
    lines = contents.split('\n')
    lines = [line for line in lines if line != ''] if ignore_blanks else lines
    return lines


# Key for wikipedia eval is question-id. Key for web eval is the (question_id, filename) tuple
def get_key_to_ground_truth(data):
    if data['Domain'] == 'Wikipedia':
        return {datum['QuestionId']: datum['Answer'] for datum in data['Data']}
    else:
        return get_qd_to_answer(data)


def get_question_doc_string(qid, doc_name):
    return '{}--{}'.format(qid, doc_name)

def get_qd_to_answer(data):
    key_to_answer = {}
    for datum in data['Data']:
        for page in datum.get('EntityPages', []) + datum.get('SearchResults', []):
            qd_tuple = get_question_doc_string(datum['QuestionId'], page['Filename'])
            key_to_answer[qd_tuple] = datum['Answer']
    return key_to_answer


def read_clean_part(datum):
    for key in ['EntityPages', 'SearchResults']:
        new_page_list = []
        for page in datum.get(key, []):
            if page['DocPartOfVerifiedEval']:
                new_page_list.append(page)
        datum[key] = new_page_list
    assert len(datum['EntityPages']) + len(datum['SearchResults']) > 0
    return datum


def read_triviaqa_data(qajson):
    data = read_json(qajson)
    # read only documents and questions that are a part of clean data set
    if data['VerifiedEval']:
        clean_data = []
        for datum in data['Data']:
            if datum['QuestionPartOfVerifiedEval']:
                if data['Domain'] == 'Web':
                    datum = read_clean_part(datum)
                clean_data.append(datum)
        data['Data'] = clean_data
    return data


def answer_index_in_document(answer, document):
    answer_list = answer['normalized_aliases']
    for answer_string_in_doc in answer_list:
        index = document.lower().find(answer_string_in_doc)
        if index != -1:
            return answer_string_in_doc, index
    return answer['NormalizedValue'], -1


# -*- coding: utf-8 -*-
""" Official evaluation script for v1.0 of the TriviaQA dataset.
Extended from the evaluation script for v1.1 of the SQuAD dataset. """
from collections import Counter
import string
import re
import sys
import argparse, pdb


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def f1_score(prediction, ground_truth):
    # pdb.set_trace()
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    # TODO, change the logic or exact match to be more relaxed -> once the answer is in the generated sentence, it is regarded as correct
    if len(prediction)==0:
        return 0
    if (normalize_answer(prediction) == normalize_answer(ground_truth))  or  (normalize_answer(ground_truth) in normalize_answer(prediction)) or (normalize_answer(prediction) in normalize_answer(ground_truth)):
        return 1
    else:
        return 0
    # return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def is_exact_match(answer_object, prediction):
    ground_truths = get_ground_truths(answer_object)
    for ground_truth in ground_truths:
        if exact_match_score(prediction, ground_truth):
            return True
    return False


def has_exact_match(ground_truths, candidates):
    for ground_truth in ground_truths:
        if ground_truth in candidates:
            return True
    return False


def get_ground_truths(answer):
    return answer['normalized_aliases'] + [normalize_answer(ans) for ans in answer.get('HumanAnswers', [])] + [
        normalize_answer(answer['normalized_value'])]


def get_oracle_score(ground_truth, predicted_answers, qid_list=None, mute=False):
    exact_match = common = 0
    if qid_list is None:
        qid_list = ground_truth.keys()
    for qid in qid_list:
        if qid not in predicted_answers:
            if not mute:
                message = 'Irrelavant question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        common += 1
        prediction = normalize_answer(predicted_answers[qid])
        ground_truths = get_ground_truths(ground_truth[qid])
        em_for_this_question = has_exact_match(ground_truths, prediction)
        exact_match += int(em_for_this_question)

    exact_match = 100.0 * exact_match / len(qid_list)

    return {'oracle_exact_match': exact_match, 'common': common, 'denominator': len(qid_list),
            'pred_len': len(predicted_answers), 'gold_len': len(ground_truth)}


def evaluate_triviaqa(ground_truth, predicted_answers, qid_list=None, mute=False):
    """
        ground_truth: a dict of question_id -> answer
        predicted_answers: a dict of question_id -> answer
    """
    f1 = exact_match = common = 0
    error_id = []
    if qid_list is None:
        qid_list = ground_truth.keys()
    is_correct = {}
    for qid in qid_list:
        if qid not in predicted_answers:
            if not mute:
                message = 'Missed question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        if qid not in ground_truth:
            if not mute:
                message = 'Irrelavant question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        # common means the question and answer both exist
        common += 1
        # "qid" -> "sunset boulevard" 
        prediction = predicted_answers[qid]
        if isinstance(ground_truth[qid], dict):
            ground_truths = get_ground_truths(ground_truth[qid])
        else:
            ground_truths = [ground_truth[qid]]
        
        # compare exact match
        em_for_this_question = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        # not exact match
        if em_for_this_question == 0 and not mute:
            error_id.append(qid)
        exact_match += em_for_this_question
        
        # compute f1 score 
        f1_for_this_question = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        f1 += f1_for_this_question
        
        is_correct[qid] = em_for_this_question

    exact_match = 100.0 * exact_match / len(qid_list)
    f1 = 100.0 * f1 / len(qid_list)

    return {'exact_match': exact_match, 'f1': f1, 'common': common, 'denominator': len(qid_list),
            'pred_len': len(predicted_answers), 'gold_len': len(ground_truth), 'error_id': error_id, 'is_correct': is_correct}
    
def evaluate_nq(ground_truth, predicted_answers, qid_list=None, mute=False):
    """
        ground_truth: a dict of question_id -> answer
        predicted_answers: a dict of question_id -> answer
    """
    f1 = exact_match = common = 0
    error_id = []
    if qid_list is None:
        qid_list = ground_truth.keys()

    is_correct = {}
    for qid in qid_list:
        if qid not in predicted_answers:
            if not mute:
                message = 'Missed question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        if qid not in ground_truth:
            if not mute:
                message = 'Irrelavant question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        # common means the question and answer both exist
        common += 1
        # "qid" -> "sunset boulevard" 
        prediction = predicted_answers[qid]
        if isinstance(ground_truth[qid], dict):
            ground_truths = get_ground_truths(ground_truth[qid])
        else:
            ground_truths = [ground_truth[qid]]
        
        # compare exact match
        for single_ground_truth in ground_truths:
            em_for_this_question = metric_max_over_ground_truths(
                exact_match_score, prediction, single_ground_truth)
            exact_match += em_for_this_question
            
        # not exact match
        if em_for_this_question == 0 and not mute:
            error_id.append(qid)
        
        # compute f1 score 
        for single_ground_truth in ground_truths:
            f1_for_this_question = metric_max_over_ground_truths(
            f1_score, prediction, single_ground_truth)
            f1 += f1_for_this_question
        
        is_correct[qid] = em_for_this_question
        
    exact_match = 100.0 * exact_match / len(qid_list)
    f1 = 100.0 * f1 / len(qid_list)

    return {'exact_match': exact_match, 'f1': f1, 'common': common, 'denominator': len(qid_list),
            'pred_len': len(predicted_answers), 'gold_len': len(ground_truth), 'error_id': error_id, 'is_correct': is_correct}


    
def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluation for TriviaQA {}'.format(expected_version))
    parser.add_argument('--dataset_file', help='Dataset file')
    parser.add_argument('--prediction_file', help='Prediction File')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    expected_version = 1.0
    args = get_args()

    dataset_json = read_triviaqa_data(args.dataset_file)
    if dataset_json['Version'] != expected_version:
        print('Evaluation expects v-{} , but got dataset with v-{}'.format(expected_version,dataset_json['Version']),
              file=sys.stderr)
    key_to_ground_truth = get_key_to_ground_truth(dataset_json)
    predictions = read_json(args.prediction_file)
    eval_dict = evaluate_triviaqa(key_to_ground_truth, predictions)
    print(eval_dict)
