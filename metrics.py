import numpy as np

def tied_rank(x):

    sorted_x=sorted(zip(x,range(len(x))))
    r=[0 for k in x]
    cur_val=sorted_x[0][0]
    last_rank=0
    for i in range(len(sorted_x)):
        if cur_val!=sorted_x[i][0]:
            cur_val=sorted_x[i][0]
            for j in range(last_rank,i):
                r[sorted_x[j][1]]=float(last_rank+1+i)/2.0
            last_rank=i
        if i==len(sorted_x)-1:
            for j in range(last_rank,i+1):
                r[sorted_x[j][1]]=float(last_rank+i+2)/2.0
    return r

def auc(y_true,y_score):
    r=tied_rank(y_score)
    num_positive=len([0 for x in y_true if x==1])
    num_negative=len(y_true)-num_positive
    sum_positive=sum([r[i] for i in range(len(r)) if y_true[i]==1])
    auc=((sum_positive-num_positive*(num_positive+1)/2.0)/
         (num_negative*num_positive))
    return auc

def logloss(y_true,y_pred,normalize=True):
    loss_array=-y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred)
    if normalize:
        return np.mean(loss_array)
    else:
        return np.sum(loss_array)
