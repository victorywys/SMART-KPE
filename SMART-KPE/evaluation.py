import os, sys

import numpy as np 

def evaluate_kps(true_ls, pred_ls):
    result = {}
    for length in [1,3,5]:
        result[f"P@{length}"] = 1e-8
        result[f"R@{length}"] = 1e-8
        result[f"F@{length}"] = 1e-8 
    for i in range(len(true_ls)):
        cur_true = set(true_ls[i])
        cur_pred = pred_ls[i]
        match_ls = []
        match_cnt = 0
        if len(cur_pred)<=0:
            match_ls = [0]
        for kp in cur_pred[:5]:
            if kp in cur_true:
                match_cnt += 1
            match_ls.append(match_cnt)
        for length in [1,3,5]:
            if length>len(match_ls):
                result[f"P@{length}"] += match_ls[-1] / len(match_ls)
                result[f"R@{length}"] += match_ls[-1] / len(cur_true)
            else:    
                result[f"P@{length}"] += match_ls[length-1] / length
                result[f"R@{length}"] += match_ls[length-1] / len(cur_true)
    for length in [1,3,5]:
        result[f"P@{length}"] /= len(true_ls)
        result[f"R@{length}"] /= len(true_ls)
        result[f"F@{length}"] = 2.0 / (1.0/result[f"P@{length}"] + 1.0/result[f"R@{length}"])
    return result    

