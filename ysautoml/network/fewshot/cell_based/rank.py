import ast
import json
import torch

from scipy import stats


SEED=0
#NAME=f'seed-{SEED}-opt5-wotrash'
#NAME=f'seed-{SEED}-opt5-wotrash-baseline-5-128-1000-balanced'
NAME=f'seed-{SEED}-opt5-wotrash-decom5-8-25-64-500-balanced' # K=2
#NAME=f'seed-{SEED}-opt5-wotrash-decom5-5-15-20-64-750-balanced' # K=3
#NAME=f'seed-{SEED}-opt5-wotrash-decom5-4-10-20-30-128-1000' # K=4

print("-*"*10)
print(NAME)
print("-*"*10)

def compute_kendall(pred_dict, ranked_gt_list):
    gt_acc_list = []
    pred_acc_list = [] 
    for arch, acc in ranked_gt_list:
        gt_acc_list += [acc]
        pred_acc_list += [pred_dict[arch]]
            
    
    gt_rank_list = list(range(1, len(ranked_gt_list)+1))
    
    pred_rank = 1
    pred_rank_list = torch.zeros(len(ranked_gt_list), dtype=torch.long)
    _,ind_list = torch.tensor(pred_acc_list).sort(descending=True)
    for ind in ind_list:
        pred_rank_list[ind] = pred_rank
        pred_rank += 1
    pred_rank_list = pred_rank_list.tolist()
    
    
    tau = stats.kendalltau(gt_rank_list, pred_rank_list)[0]
    print(f"Kendall tau (rank) : {tau:.3f}")
    
    tau = stats.kendalltau(gt_acc_list, pred_acc_list)[0]
    print(f"Kendall tau (real-valued) : {tau:.3f}")
    

def tostr(nodes):
    strings = []
    for node_info in nodes:
        string = '|'.join([x[0]+'~{:}'.format(x[1]) for x in node_info])
        string = '|{:}|'.format(string)
        strings.append( string )
    return '+'.join(strings)


gt_dict = json.load(open('./SuperNet/logs/gt_c10', 'r'))
#tmp_dict = json.load(open('./SuperNet/logs/OneShot', 'r'))
#tmp_dict = json.load(open('./SuperNet/logs/oneshot_test', 'r'))

tmp0 = json.load(open(f'./SuperNet/logs/{NAME}_0', 'r'))
tmp1 = json.load(open(f'./SuperNet/logs/{NAME}_1', 'r'))
tmp2 = json.load(open(f'./SuperNet/logs/{NAME}_2', 'r'))
tmp3 = json.load(open(f'./SuperNet/logs/{NAME}_3', 'r'))
tmp4 = json.load(open(f'./SuperNet/logs/{NAME}_4', 'r'))
tmp_dict = {**tmp0, **tmp1, **tmp2, **tmp3, **tmp4}

pred_dict = {}
for k,v in tmp_dict.items():
    pred_dict[tostr(ast.literal_eval(k))] = v
del tmp0, tmp1, tmp2, tmp3, tmp4, tmp_dict
#del tmp_dict

assert len(gt_dict) == len(pred_dict)

ranked_gt_list = sorted(gt_dict.items(), key=lambda d: d[1], reverse=True)
#ranked_pred_list = sorted(pred_dict.items(), key=lambda d: d[1], reverse=True)
#import pdb; pdb.set_trace()

TOP1 = int(len(pred_dict)*0.01)
TOP50 = int(len(pred_dict)*0.5)

print("For All")
compute_kendall(pred_dict, ranked_gt_list)
print()
print("For Top-1%")
compute_kendall(pred_dict, ranked_gt_list[:TOP1])
print()
print("For Top-50%")
compute_kendall(pred_dict, ranked_gt_list[:TOP50])
print()
print("For Bottom-50%")
compute_kendall(pred_dict, ranked_gt_list[TOP50:])

