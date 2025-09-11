import json
import torch

from scipy import stats


NAME='seed-0-last_arch320'
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
ranked_gt_list = sorted(gt_dict.items(), key=lambda d: d[1], reverse=True)
new_gt_list = []
tacc_list = []
for arch, acc in ranked_gt_list[::48]:
    if acc in tacc_list:
        continue
    else:
        tacc_list += [acc]
        new_gt_list += [(arch, acc)]

pred_dict = json.load(open(f'./SuperNet/logs/{NAME}', 'r'))

assert len(new_gt_list) == len(pred_dict)

TOP1 = int(len(pred_dict)*0.01)
TOP50 = int(len(pred_dict)*0.5)

print(f"For All {len(pred_dict)}")
compute_kendall(pred_dict, new_gt_list)
print()
print(f"For Top-1% {TOP1}")
compute_kendall(pred_dict, new_gt_list[:TOP1])
print()
print(f"For Top-50% {TOP50}")
compute_kendall(pred_dict, new_gt_list[:TOP50])
print()
print(f"For Bottom-50% {TOP50}")
compute_kendall(pred_dict, new_gt_list[TOP50:])

