import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import metrics
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#data processing
train_data = np.loadtxt('letter_train',delimiter=' ',dtype=str)
test_data = np.loadtxt('letter_test',delimiter=' ',dtype=str)

train_label = train_data[:,0].astype(np.int)
test_label = test_data[:,0].astype(np.int)

train = train_data[:,1:-1]
test = test_data[:,1:-1]
train_features = []
test_features = []

for features in train:
    x = [np.fromstring(feature,dtype=np.float,sep=':')[1] for feature in features]
    train_features.append(x)
#print(train_features)

for features in test:
    x = [np.fromstring(feature,dtype=np.float,sep=':')[1] for feature in features]
    test_features.append(x)
#print(test_features)

#decision tree

clf1 = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 5, random_state = 0)
clf2 = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 10, random_state = 0)
clf3 = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 15, random_state = 0)
clf4 = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 20, random_state = 0)
clf5 = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 25, random_state = 0)
clf6 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 0)
clf7 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 10, random_state = 0)
clf8 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 15, random_state = 0)
clf9 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 20, random_state = 0)
clf10 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 25, random_state = 0)

model1 = clf1.fit(train_features, train_label)
model2 = clf2.fit(train_features, train_label)
model3 = clf3.fit(train_features, train_label)
model4 = clf4.fit(train_features, train_label)
model5 = clf5.fit(train_features, train_label)
model6 = clf6.fit(train_features, train_label)
model7 = clf7.fit(train_features, train_label)
model8 = clf8.fit(train_features, train_label)
model9 = clf9.fit(train_features, train_label)
model10 = clf10.fit(train_features, train_label)

predict_results1 = model1.predict(test_features)
predict_results2 = model2.predict(test_features)
predict_results3 = model3.predict(test_features)
predict_results4 = model4.predict(test_features)
predict_results5 = model5.predict(test_features)
predict_results6 = model6.predict(test_features)
predict_results7 = model7.predict(test_features)
predict_results8 = model8.predict(test_features)
predict_results9 = model9.predict(test_features)
predict_results10 = model10.predict(test_features)

accuracy_score(test_label, predict_results10) 
#0.3694 0.7142 0.832 0.866 0.8764 
#0.5014 0.7974 0.8728 0.8712 0.8712

metrics.precision_score(test_label, predict_results10, average='macro') 
#0.3924 0.7639 0.8428 0.8680 0.8754
#0.5610 0.8036 0.8732 0.8716 0.8716

metrics.recall_score(test_label, predict_results10, average='macro') 
#0.3685 0.7172 0.8326 0.866 0.8748
#0.5014 0.7973 0.8735 0.8716 0.8716

metrics.f1_score(test_label, predict_results10, average='weighted') 
#0.3256 0.7275 0.8344 0.8663 0.8747
#0.4968 0.7980 0.8728 0.8712 0.8712

start = time.time()

#change the model here
clf10 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 25, random_state = 0)
model10 = clf10.fit(train_features, train_label)

end = time.time()
running_time = end-start
print('time cost : %.5f sec' %running_time)
#0.07224sec 0.11699sec 0.11555sec 0.16407sec 0.12257sec
#0.07398sec 0.10805sec 0.12334sec 0.12009sec 0.13632sec


#KNN
clfK1 = KNeighborsClassifier(n_neighbors=3, weights='uniform', leaf_size=30)
clfK2 = KNeighborsClassifier(n_neighbors=3, weights='distance', leaf_size=30)
clfK3 = KNeighborsClassifier(n_neighbors=3, weights='uniform', leaf_size=15)

modelK1 = clfK1.fit(train_features, train_label)
modelK2 = clfK2.fit(train_features, train_label)
modelK3 = clfK3.fit(train_features, train_label)

predict_resultsK1 = modelK1.predict(test_features)
predict_resultsK2 = modelK2.predict(test_features)
predict_resultsK3 = modelK3.predict(test_features)

accuracy_score(test_label, predict_resultsK1)
#0.952 0.9562 0.9518

metrics.precision_score(test_label, predict_resultsK3, average='macro')
#0.9530 0.9563 0.9528

metrics.recall_score(test_label, predict_resultsK3, average='macro')
#0.9521 0.9564 0.9519

metrics.f1_score(test_label, predict_resultsK3, average='weighted')
#0.9521 0.9562 0.9519

startK = time.time()

#clfK1 = KNeighborsClassifier(n_neighbors=3, weights='uniform', leaf_size=30)
#modelK1 = clfK1.fit(train_features, train_label)
#clfK2 = KNeighborsClassifier(n_neighbors=3, weights='distance', leaf_size=30)
#modelK2 = clfK2.fit(train_features, train_label)
clfK3 = KNeighborsClassifier(n_neighbors=3, weights='uniform', leaf_size=15)
modelK3 = clfK3.fit(train_features, train_label)

endK = time.time()
running_timeK = endK-startK
print('time cost : %.5f sec' %running_timeK)

#0.18628 0.17331 0.11600


#random forest
clfRF1 = RandomForestClassifier(criterion = 'gini', max_depth = 10, n_estimators=10)
clfRF2 = RandomForestClassifier(criterion = 'gini', max_depth = 20, n_estimators=10)
clfRF3 = RandomForestClassifier(criterion = 'gini', max_depth = 20, n_estimators=100)

modelRF1 = clfRF1.fit(train_features, train_label)
modelRF2 = clfRF2.fit(train_features, train_label)
modelRF3 = clfRF3.fit(train_features, train_label)

predict_resultsRF1 = modelRF1.predict(test_features)
predict_resultsRF2 = modelRF2.predict(test_features)
predict_resultsRF3 = modelRF3.predict(test_features)

accuracy_score(test_label, predict_resultsRF3)
#0.8198 0.9352 0.9612

metrics.precision_score(test_label, predict_resultsRF3, average='macro')
#0.8279 0.9369 0.9624

metrics.recall_score(test_label, predict_resultsRF3, average='macro')
#0.8206 0.9344 0.9623

metrics.f1_score(test_label, predict_resultsRF3, average='weighted')
#0.8237 0.9340 0.9616

startRF = time.time()

#clfRF1 = RandomForestClassifier(criterion = 'gini', max_depth = 10, n_estimators=10)
#modelRF1 = clfRF1.fit(train_features, train_label)
#clfRF2 = RandomForestClassifier(criterion = 'gini', max_depth = 20, n_estimators=10)
#modelRF2 = clfRF2.fit(train_features, train_label)
clfRF3 = RandomForestClassifier(criterion = 'gini', max_depth = 20, n_estimators=100)
modelRF3 = clfRF3.fit(train_features, train_label)

endRF = time.time()
running_timeRF = endRF-startRF
print('time cost : %.5f sec' %running_timeRF)

#0.19306 0.22626 1.88783



