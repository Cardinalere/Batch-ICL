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
from nltk.translate.bleu_score import sentence_bleu
import time

TASKS = ['WMT']

alpha = 0
choices = ["A", "B", "C", "D"]

device = torch.device("cuda")
def compute_metric_SST2(output_filename,ls):
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
            bleu_score = sentence_bleu([gold], pred)
            acc += bleu_score
        #acc /= len(pred_answers)
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
            bleu_score = sentence_bleu([gold],pred)
            acc += bleu_score
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

def format_example(df, idx, include_answer=True):
    prompt = "Sentence:"+df.iloc[idx, 0]
    prompt += "\nAnswer:"
    if include_answer:
        prompt += "{}\n\n".format(df.iloc[idx, 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "Translate the given sentence.\n\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt




def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    input_tokens = {k:input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)

    return input_tokens

def load(ckpt_dir,model_type):
    n_gpus = 1
    hub_token = "Your token here"
    if model_type == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(
            ckpt_dir, token=hub_token,padding_side="left"
        )
        model = LlamaForCausalBatchICLLM.from_pretrained(ckpt_dir,low_cpu_mem_usage = True, torch_dtype=torch.float32).to(device)

        

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
            outp = outp["hidden_before_mlp"]
            #outp = outp["hidden_states"]
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
        if max_l == 1:
            ans.append(x)
            continue
        nl = -1 
        st = -1
        for i in range(len(x)):
            if (x[i]=='\n') :
                nl = i
                break

        ans_x = x[:nl]
        ans.append(ans_x)
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
    #ans = tokenizer.decode(pos_)
    #print(ans)
    ans = tokenizer.convert_ids_to_tokens(pos_)
    return ans

def batch_infer(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    tl = 0
    ans_r = []
    u = 0
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        ins = encode_inputs["input_ids"]
        if tl==0:
            print(tokenizer.convert_ids_to_tokens(ins[0]))
            tl+=1
        outputs = model.generate(**encode_inputs, do_sample=False,max_new_tokens=256, pad_token_id=tokenizer.pad_token_id)
        p = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        for i in range(len(p)):
            x = p[i]
            u_l = len(prompts[u+i])
            x = x[u_l:]
            nl = 0 
            st = -1
            for i in range(len(x)):
                if (st==-1) and (x[i-1]=='#') and (x[i-2]=='#') and (x[i-3]=='#') and (x[i-4]=='#'):
                    st=i+1
                    i=st+1
                    continue
                if (st==-1):
                    continue
                if (st!=-1) and (x[i]<'0') or (x[i]>'9') and (x[i]!='#'):
                    nl = i
                    break
            ans_x = x[st:nl]
            ans_r.append(ans_x)
        u += len(ins)

    return ans_r


def batch_infer_num(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    tl = 0
    ans_r = []
    u = 0
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        ins = encode_inputs["input_ids"]
        if tl==0:
            print(tokenizer.convert_ids_to_tokens(ins[0]))
            tl+=1
        outputs = model.generate(**encode_inputs, do_sample=False,max_new_tokens=70, pad_token_id=tokenizer.pad_token_id)
        p = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        for i in range(len(p)):
            x = p[i]
            u_l = len(prompts[u+i])
            x = x[u_l:]
            nl = -1 
            st = -1
            
            for i in range(len(x)):
                if (x[i]=='\n') :
                    nl = i
                    break
                          
            ans_x = x[:nl]
            ans_r.append(ans_x)
        u += len(ins)

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
    for task in range(1):
        roun += 1
        print("Now_Round:"+str(roun))
        acc = []
        acc_u = []
        acc_nbce = []
        k = args.ntrain
        for no_ in range(1):
            print("Now_test:"+str(no_))
            records = []
            records_try = []
            run_results = {}
            dev_df = pd.read_csv('data/WMT/shot/'+str(k)+'/dev_'+str(no_)+".csv", header=None)[:args.ntrain]
            test_df = pd.read_csv("data/WMT/test.csv", header=None)
            for i in range(test_df.shape[0]):
                k = args.ntrain
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, k)
                prompt = train_prompt + prompt_end
                while len(tokenizer.tokenize(prompt)) + 1> 4096: # bos token
                    prompt_split = prompt.split("\n\n")
                    prompt_split.pop(1)
                    prompt = '\n\n'.join(prompt_split)
                label = str(test_df.iloc[i, 1])
                records.append({'prompt':prompt, 'answer':label})
            
            pred_answers = batch_infer_num(model, tokenizer, [record['prompt'] for record in records])
            gold_answers = [record['answer'] for record in records]
            run_results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
            with open(output_filename, 'w') as f:
                json.dump(run_results, f, ensure_ascii=False, indent=2)
    
            acc_ = compute_metric(output_filename)

            acc.append(acc_)
            
            tot_l = get_tot_l(args.param_size)
            ls = []
            st_l = 0
            for ly in range(st_l,tot_l+1):
                print("INs_Layer:"+str(ly))
                k = args.ntrain
                if ly==st_l:
                    records = []
                    records_1shots = []
                    for i in range(test_df.shape[0]):
                        k = args.ntrain
                        prompt_end = format_example(test_df, i, include_answer=False)
                        prop = "Translate the given sentence.\n\n"
                        prompt = prop + prompt_end
                        while len(tokenizer.tokenize(prompt)) + 1> 4096: 
                            prompt_split = prompt.split("\n\n")
                            prompt_split.pop(1)
                            prompt = '\n\n'.join(prompt_split)
                        label = str(test_df.iloc[i, 1])
                        records.append({'prompt':prompt, 'answer':label})
            
                        for j in range(k):
                            prop = "Translate the given sentence.\n\n"
                            only_ques = prop
                            prop += format_example(dev_df, j)
                            prop += prompt_end
                            only_ques += prompt_end
                            prompt = prop
                            while len(tokenizer.tokenize(prompt)) + 1> 4096: # bos token
                                prompt_split = prompt.split("\n\n")
                                prompt_split.pop(1)
                                prompt = '\n\n'.join(prompt_split)
                            label = str(test_df.iloc[i, 1])
                            records_1shots.append({'prompt':prompt, 'que_id':i, 'answer':label})
                if ly == st_l:
                    run_results = {}
                    hiddens,logits = batch_get_hidden(model,tokenizer,[record['prompt'] for record in records_1shots],k,test_df.shape[0],pos=-2)
                    acc_nbce.append(0)
                    run_results = {}
                    hiddens_ = hiddens
                else:
                    hiddens = hiddens_
                if ly == tot_l:
                    ins_ly = -1
                else:
                    ins_ly = ly
        
                ls.append(str(ly))
                pred_answers = batch_infer_with_hiddens(model, tokenizer, [record['prompt'] for record in records],ins_ly,-2,hiddens,gen_len=70,insert_le=2)
                gold_answers = [record['answer'] for record in records]
                print(pred_answers[:50])
                run_results[str(ly)] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
            with open(output_filename, 'w') as f:
                json.dump(run_results, f, ensure_ascii=False, indent=2)
    
            acc_u_ = compute_metric_SST2(output_filename,ls)
            acc_u.append(acc_u_)
    
            end_time = time.time()
            print("total run time %.2f" % (end_time - start_time))
        acc_u = np.array(acc_u)
        acc = np.array(acc)
        acc_mean = np.mean(acc)
        acc_u_mean = np.mean(acc_u)
        acc_var = np.var(acc)
        acc_u_var = np.var(acc_u)
        print("BaseLine_Mean_Acc:"+str(acc_mean))
        print("BatchICL_Mean_Acc:"+str(acc_u_mean))
        print("BaseLine_Var_Acc:"+str(acc_var))
        print("BatchICL_Var_Acc:"+str(acc_u_var))

        
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--param_size', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--ntrain', type=int, default=4)
    args = parser.parse_args()
    
    main(args.ckpt_dir, args.param_size, args.model_type)

