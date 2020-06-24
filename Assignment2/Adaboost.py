import numpy as np
import random

def weak_classifier(data,Dt):
    m = data.shape[0]
    pred = []
    threshold = None
    mark = None
    min_err = np.inf
    for j in range(m):
        pred_temp = []
        sub_mark = None
        lsum = np.sum(data[:j,1])
        rsum = np.sum(data[j:,1])
        if lsum < rsum:
            sub_mark = -1.
            pred_temp.extend([-1.]*(j))
            pred_temp.extend([1.]*(m-j))
        else:
            sub_mark = 1.
            pred_temp.extend([1.]*(j))
            pred_temp.extend([-1.]*(m-j))
        err = np.sum(1*(data[:,1]!=pred_temp)*Dt)
        if err < min_err:
            min_err = err
            threshold = (data[:,0][j-1]+data[:,0][j])/2
            mark = sub_mark
            pred = pred_temp[:]
    model = [threshold,mark,min_err]
    return model,pred

def adaboost(data):
    models = []
    m = data.shape[0]
    D = np.zeros(m)+1.0/m
    T = 3 #the number of weak classfier
#    T=5
    y = data[:,-1]
    
    for t in range(T):
        Dt = D[:]

#        sampling        
#        a=random.sample(list(data),10)
#        data=np.array(a)
#        print(data)   

        model,y_ = weak_classifier(data,Dt)
    
#        print(model)
        errt = model[-1]
        alpha = 0.5*np.log((1-errt)/errt)
        Zt = np.sum([Dt[i]*np.exp(-alpha*y[i]*y_[i]) for i in range(m)])
        D = np.array([Dt[i]*np.exp(-alpha*y[i]*y_[i]) for i in range(m)])/Zt
        models.append([model,alpha])

    return models

data=np.array([[0,1],[1,1],[2,1],[3,-1],[4,-1],[5,-1],[6,1],[7,1],[8,1],[9,-1]],dtype=np.float32)
model=adaboost(data)
print(model)
