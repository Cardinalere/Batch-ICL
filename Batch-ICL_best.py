import argparse
import json
import os
import time

import pandas as pd
import tensor_parallel as tp
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM,LlamaForCausalBatchICLLM
import transformers
import numpy as np



TASKS = [
        'SST2',
        'RTE',
        'AGnews',
        'QNLI',
        'mrpc'
        ]

alpha = 0

device = torch.device("cuda")
def compute_metric_layers(output_filename,ls):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    max_acc = 0
    lay = 0
    acc_s = []
    for tk in ls:
        acc = 0
        pred_answers = run_results[tk]['pred_answers']
        gold_answers = run_results[tk]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%2s: %.4f" % (tk,acc/len(gold_answers)))
        
        acc_s.append(acc*1.0/len(gold_answers))
        max_acc = max(acc,max_acc)
        if max_acc == acc:
            lay = tk
        total_num = len(gold_answers)
    print("ACC-MAX: %.4f" % (max_acc/total_num))
    acc_t = torch.Tensor(acc_s)
    print(lay)
    return max_acc/total_num

def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))
    return total_acc/total_num


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, task , include_answer=True):
    if task == 'SST2':
        prompt = "Review:"+df.iloc[idx, 0]
        prompt += "\nAnswer:"
        if include_answer:
            prompt += "{}\n\n".format(df.iloc[idx, 1])
        return prompt
    if task == 'AGnews':
        prompt = "Title:"+df.iloc[idx, 1]+"\nDescription:"+df.iloc[idx, 2]
        prompt += "\nAnswer:"
        if include_answer:
            prompt += "{}\n\n".format(df.iloc[idx, 0])
        return prompt
    if task == 'RTE':
        prompt = "Premise:"+df.iloc[idx, 0]+"\nHypothesis:"+df.iloc[idx, 1]
        prompt += "\nAnswer:"
        if include_answer:
            prompt += "{}\n\n".format(df.iloc[idx, 2])
        return prompt
    if task == 'mrpc':
        prompt = "Sentence 1:"+df.iloc[idx, 0]+"\nSentence 2:"+df.iloc[idx, 1]
        prompt += "\nAnswer:"
        if include_answer:
            prompt += "{}\n\n".format(df.iloc[idx, 2])
        return prompt
    if task == 'QNLI':
        prompt = "Question:"+df.iloc[idx, 0]+"\nParagraph:"+df.iloc[idx, 1]
        prompt += "\nAnswer:"
        if include_answer:
            prompt += "{}\n\n".format(df.iloc[idx, 2])
        return prompt

def get_p(task):
    if task == 'SST2':
        prompt = "The following are multiple film reviews with answers(← or →).\n\n"
        return prompt
    if task == 'AGnews':
        prompt = "Classify the news articles into the categories of 1, 2, 3, or 4.\n\n"
        return prompt
    if task == 'RTE':
        prompt = "Determine whether the hypotheses made based on the premises below are ↑ or ↓.\n\n"
        return prompt
    if task == 'mrpc':
        prompt = "Assess if each pair reflects a semantic equivalence relationship. Use ← or → to indicate the answers.\n\n"
        return prompt
    if task == 'QNLI':
        prompt = "Please determine whether the paragraph contains the answer to the corresponding question. Use ↑ or ↓ to indicate the answers.\n\n"
        return prompt

def gen_prompt(train_df, task, k=-1):
    prompt = get_p(task)
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i,task=task)
    return prompt



def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    input_tokens = {k:input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)

    return input_tokens

def load(ckpt_dir,model_type):
    hub_token = "Your token here" 
    if model_type == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(
            ckpt_dir, token=hub_token,padding_side="left"
        )
        model = LlamaForCausalBatchICLLM.from_pretrained(ckpt_dir,low_cpu_mem_usage = True, torch_dtype=torch.float16).to(device)

        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1

    model = model.eval()

    return model, tokenizer

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_get_hidden(model, tokenizer, prompts,k,n,pos):
    batch_size = k
    answers = []
    tim = 0
    ptt = 0
    logs = 0
    logit = 0
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        if ptt == 0:
            ptt = 1
            ins = encode_inputs["input_ids"]
            print(tokenizer.convert_ids_to_tokens(ins[0]))
        with torch.no_grad():
            outp = model(**encode_inputs, output_hidden_states = True, return_dict = True,out_attn=True,idea=0)#["hidden_states"]
            out_logits = outp["logits"][:,-1,:]
            #outp = outp["hidden_before_mlp"]
            outp = outp["hidden_states"]
        #print(outp[0].shape)
        l_s = outp[0][:,pos:,:].unsqueeze(0)
        for i in range(len(outp)):
            out_s = outp[i][:,pos:,:].unsqueeze(0)
            if i==0:
                continue
            l_s = torch.cat((l_s,out_s),dim=0)
        #print("ls:"+str(l_s.shape))
        l_s = torch.transpose(l_s,0,1)
        hidden_j = l_s[0]
        log_j = out_logits[0]
        for j in range(k):
            if j==0:
                continue
            hidden_j = torch.add(hidden_j,l_s[j])
            log_j = torch.add(log_j,out_logits[j])
        hidden_j = hidden_j/k
        log_j = log_j/k
        hidden_j = hidden_j.unsqueeze(0)
        log_j = log_j.unsqueeze(0)
        if tim == 0:
            tim += 1
            hiddens = hidden_j
            logit = log_j
        else:
            hiddens = torch.cat((hiddens,hidden_j),dim=0)
            logit = torch.cat((logit,log_j),dim=0)
    return hiddens,logit

def gen_with_h(model, tokenizer, batch_input,insert_layer,insert_positions,hiddens,max_l,insert_le=1):
    ans = []
    pre_l = 0
    encode_inputs = prepare_input(tokenizer, batch_input)
    ins = encode_inputs["input_ids"]
    bs = ins.shape[0]
    att = encode_inputs["attention_mask"]
    ct_a = torch.Tensor([1]*bs).to(device)
    ct_a = ct_a.unsqueeze(1)
    for _ in range(max_l):
        if pre_l == 0:
            pre_l = ins.shape[1]
        ins_p = torch.LongTensor([insert_positions]*bs).to(device)
        ins_le = torch.LongTensor([insert_le]*bs).to(device)

        insert_positions -= 1

        with torch.no_grad():
            outp = model(input_ids = ins,attention_mask=att,ins_attn=True, insert_layer=insert_layer,insert_pos=ins_p,insert_hiddens = hiddens,insert_len = ins_le,idea=1,alpha = alpha)["logits"]
        outs = outp[:,-1,:]
        score_ = outs.clone().detach().cpu().numpy()
        pos_ = torch.LongTensor([np.argmax(score_[i]) for i in range(bs)]).to(device)
        pos_ = pos_.unsqueeze(1)
        ins = torch.cat((ins,pos_),dim=1)
        att = torch.cat((att,ct_a),dim=1)
        #print(as_p)
    ps = tokenizer.batch_decode(ins[:,pre_l:], skip_special_tokens=True)

    for x in ps:
        #print(x)
        if max_l == 1:
            ans.append(x)
            continue
        if x[-1] == "\n":
            ans.append(x[:-1])
        else:
            ans.append(x)
    return ans
        
        
        

def batch_infer_with_hiddens(model, tokenizer, prompts,insert_layer,insert_positions,hiddens,gen_len,insert_le=1):
    batch_size = 8
    answers = []
    u = 0
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        ins = encode_inputs["input_ids"]
        bs = ins.shape[0]
        ins_l = torch.LongTensor([insert_layer]*bs).to(device)
        ins_p = torch.LongTensor([insert_positions]*bs).to(device)
        ins_hiddens = hiddens[u:u+bs].to(device)
        u += bs
        ins = encode_inputs["input_ids"]
        o = gen_with_h(model,tokenizer,batch_input,ins_l,insert_positions,ins_hiddens,max_l=gen_len,insert_le=insert_le)
        #print(pos_)
        answers.extend(o)
    answers = [answer for answer in answers]
    return answers

def batch_NBCE(tokenizer, logits):
    batch_size = 8
    answers = []
    logs = logits.detach().cpu().numpy()
    pos_ = np.argmax(logs,axis=1).tolist()
    print(len(pos_))

    ans = tokenizer.convert_ids_to_tokens(pos_)
    return ans

def batch_infer(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    tl = 0
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        ins = encode_inputs["input_ids"]
        if tl==0:
            print(tokenizer.convert_ids_to_tokens(ins[0]))
            tl+=1
        outputs = model.generate(**encode_inputs, max_new_tokens=3, pad_token_id=tokenizer.pad_token_id)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ans_r = []
    for answer in answers:
        if answer[-1]=="\n":
            pl = answer[-9:-1]
        else:
            pl = answer[-8:]
        ans_r.append(pl)
    return ans_r

def batch_infer_label(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    tl = 0
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        ins = encode_inputs["input_ids"]
        if tl==0:
            print(tokenizer.convert_ids_to_tokens(ins[0]))
            tl+=1

        outputs = model.generate(**encode_inputs, max_new_tokens=1, do_sample = False, pad_token_id=tokenizer.pad_token_id)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ans = [answer[-1] for answer in answers]
    return ans


def get_tot_l(ps):
    if int(ps)==7:
        return 32
    if int(ps)==13:
        return 40
    return 80

def main(ckpt_dir: str,param_size: str, model_type: str):
    
    run_results = {}
    run_trys = {}
    output_filename = 'run_results_%s_%sb.json' % (model_type, param_size)
    
    model, tokenizer = load(ckpt_dir,model_type)
    start_time = time.time()
    roun = -1
    for task in TASKS:
        lab_pos = -1
        if task=='AGnews':
            lab_pos = 0
        roun += 1
        print("Now_Task:"+task)
        acc_u = []
        acc = []
        k = args.ntrain
        for no_ in range(10):
            print("Now_test:"+str(no_))
            records = []
            records_try = []
            run_results = {}
            dev_df = pd.read_csv('data/'+task+'/shot/'+str(k)+'/dev_'+str(no_)+".csv", header=None)[:args.ntrain]
            test_df = pd.read_csv('data/'+task+'/test.csv', header=None)
            for i in range(test_df.shape[0]):
                k = args.ntrain
                prompt_end = format_example(test_df, i, task,include_answer=False)
                train_prompt = gen_prompt(dev_df, task,k)
                prompt = train_prompt + prompt_end
                while len(tokenizer.tokenize(prompt)) + 1> 4096: # bos token
                    prompt_split = prompt.split("\n\n")
                    prompt_split.pop(1)
                    prompt = '\n\n'.join(prompt_split)
                label = str(test_df.iloc[i, lab_pos])
                records.append({'prompt':prompt, 'answer':label})
            pred_answers = batch_infer_label(model, tokenizer, [record['prompt'] for record in records])
            gold_answers = [record['answer'] for record in records]
            run_results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
            with open(output_filename, 'w') as f:
                json.dump(run_results, f, ensure_ascii=False, indent=2)
    
            acc_ = compute_metric(output_filename)
            acc.append(acc_)
            
            
            tot_l = get_tot_l(args.param_size)
            ls = []
            for ly in range(0,tot_l):
                print("INs_Layer:"+str(ly))
                k = args.ntrain
                if ly==0:
                    records = []
                    records_1shots = []
                    for i in range(test_df.shape[0]):
                        k = args.ntrain
                        prompt_end = format_example(test_df, i, task,include_answer=False)
                        prop = get_p(task)
                        prompt = prop + prompt_end
                        while len(tokenizer.tokenize(prompt)) + 1> 4096: # bos token
                            prompt_split = prompt.split("\n\n")
                            prompt_split.pop(1)
                            prompt = '\n\n'.join(prompt_split)
                        label = str(test_df.iloc[i, lab_pos])
                        records.append({'prompt':prompt, 'answer':label})
            
                        for j in range(k):
                            prop = get_p(task)
                            only_ques = prop
                            prop += format_example(dev_df, j,task)
                            prop += prompt_end
                            only_ques += prompt_end
                            prompt = prop
                            while len(tokenizer.tokenize(prompt)) + 1> 4096: # bos token
                                prompt_split = prompt.split("\n\n")
                                prompt_split.pop(1)
                                prompt = '\n\n'.join(prompt_split)
                            label = str(test_df.iloc[i, lab_pos])
                            records_1shots.append({'prompt':prompt, 'que_id':i, 'answer':label})
                if ly == 0:
                    hiddens,logits = batch_get_hidden(model,tokenizer,[record['prompt'] for record in records_1shots],k,test_df.shape[0],pos=-2)
                    run_results = {}
                    hiddens_ = hiddens
                else:
                    hiddens = hiddens_
                if ly == tot_l:
                    ins_ly = -1
                else:
                    ins_ly = ly
        
                ls.append(str(ly))
                pred_answers = batch_infer_with_hiddens(model, tokenizer, [record['prompt'] for record in records],ins_ly,-2,hiddens,gen_len=1,insert_le=2)
                gold_answers = [record['answer'] for record in records]
                print(pred_answers[:50])
                run_results[str(ly)] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
            with open(output_filename, 'w') as f:
                json.dump(run_results, f, ensure_ascii=False, indent=2)
    
            acc_u_ = compute_metric_layers(output_filename,ls)
            acc_u.append(acc_u_)
    
            end_time = time.time()
            print("total run time %.2f" % (end_time - start_time))
        acc_u = np.array(acc_u)
        acc_u_mean = np.mean(acc_u)
        acc_u_var = np.var(acc_u)
        acc = np.array(acc)
        acc_mean = np.mean(acc)
        acc_var = np.var(acc)

        print("BaseLine_Mean_Acc:"+str(acc_mean))
        print("BaseLine_Var_acc:"+str(acc_var))
        print("BatchICL_Mean_u_Acc:"+str(acc_u_mean))
        print("BatchICL_Var_u_Acc:"+str(acc_u_var))
        
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--param_size', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--ntrain', type=int, default=4)
    args = parser.parse_args()
    
    main(args.ckpt_dir, args.param_size, args.model_type)