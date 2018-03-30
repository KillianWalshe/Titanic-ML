import pandas as pd
import numpy as np
titanic_survival= pd.read_csv('Datafiles/titanic_survival.csv')
#print(titanic_survival.head(5))
age_null=pd.isnull(titanic_survival["age"])
age_null_true=titanic_survival[age_null]
age_null_count=len(age_null_true)
#print(age_null_count)
## might change age to <18 and hae 1/0 as answer 
new_titanic = titanic_survival[['age','pclass','sex','survived']].copy()
#print(len(new_titanic))
new_titanic = new_titanic.dropna(axis=0)
#print(len(new_titanic))

#print(new_titanic.head(10))

def gender_group(row):
    gender = row['sex']
    if gender=='male':
        return(1)
    else:
        return(0)
gender = new_titanic.apply(gender_group, axis=1)

def sur_group(row):
    sur = row['survived']
    if sur==1.0:
        return(1)
    else:
        return(0)
new_titanic["survived"] = new_titanic.apply(sur_group, axis=1)

def pclass_group(row):
    pclass = row['pclass']
    if pclass==1.0:
        return(1)
    elif pclass==2.0:
        return(2)
    else:
        return(3)
new_titanic["pclass"] = new_titanic.apply(pclass_group, axis=1)

#print(gender)
new_titanic["sex"]=gender

#def age_group(row):
#    age=row["age"]
#    if age<=10:
#        return(5)
#    elif 10<age<18:
#        return(15)
#    elif 18<=age<30:
#        return(25)
#    elif 30<=age<50:
#        return(40)
#    else:
#        return(70)
#age_labels=titanic_survival.apply(age_group, axis=1)
#new_titanic["age"]=age_labels

new_titanic=new_titanic[['age','sex','pclass','survived']]
#print(new_titanic.head(10))
#print(new_titanic['pclass'].unique())

X=new_titanic.loc[:,['age','sex','pclass']]
y=new_titanic.loc[:,'survived']
#print(X)
#print(y)
#print(new_titanic.head(10))
#new_titanic=new_titanic.values
#print(new_titanic.dtype)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)    #set test size to 1/3 of total data

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train, y_train)
y_pred = ppn.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

X_plot=X.values
y_plot=y.values
X_test_plot=X_test.values
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
#fig, ax = plt.subplots()
#value = 1
#width = 0.1
#plot_decision_regions(X, y, clf=ppn,
#                      filler_feature_values={2: value},
#                      filler_feature_ranges={2: width},
#                      legend=1, ax=ax)
#ax.set_xlabel('age')
#ax.set_ylabel('pclass')
#ax.set_title('sex = {}'.format(value))

# Adding axes annotations
#fig.suptitle('SVM on make_blobs')
#plt.show()



fig, axarr = plt.subplots(2, 2, figsize=(10,8), sharex=True, sharey=True)
values = [1, 2, 3]
width = 0.1
for value, ax in zip(values, axarr.flat):
    plot_decision_regions(X_plot, y_plot, clf=ppn,X_highlight=X_test_plot,
                          filler_feature_values={2: value},
                          filler_feature_ranges={2: width},
                          legend=2, ax=ax)
    ax.set_xlabel('age - 1 = adult')
    ax.set_ylabel('sex - 1 = male')
    ax.set_title('pclass = {}'.format(value))

# Adding axes annotations
fig.suptitle('SVM on make_blobs')
plt.show()


from sklearn.svm import SVC #Support Vector Classification
accuracy=0.0
BestGamma=0.0
Bestaccuracy=0.0
BestC=0.0
Grange=np.arange(0.01,0.11,0.01)
Crange=np.arange(0.5,1.1,0.1)

for Gamma in Grange:
    for Cnum in Crange:
        svm = SVC(kernel='rbf', random_state=0, gamma=Gamma, C=Cnum)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
    #    print('Gamma:%f Misclassified samples: %d' % (Gamma, (y_test != y_pred).sum()))
        print('Gamma: %f C: %f Accuracy: %.2f' % (Gamma,Cnum, accuracy_score(y_test, y_pred)))
        if(accuracy_score(y_test, y_pred)>Bestaccuracy):
            X_plot=X.values
            y_plot=y.values
            X_test_plot=X_test.values
            BestGamma=Gamma
            Bestaccuracy= accuracy_score(y_test, y_pred)
            BestC=Cnum

print("Best Gamma = %f , Best C = %f  with accuracy = %f"% (BestGamma,BestC,Bestaccuracy))
fig, axarr = plt.subplots(2, 2, figsize=(10,8), sharex=True, sharey=True)
values = [1, 2, 3]
width = 0.1
for value, ax in zip(values, axarr.flat):
    plot_decision_regions(X_plot, y_plot, clf=svm,X_highlight=X_test_plot,
                                  filler_feature_values={2: value},
                                  filler_feature_ranges={2: width},
                                  legend=2, ax=ax)
    ax.set_xlabel('age - 1 = adult')
    ax.set_ylabel('sex - 1 = male')
    ax.set_title('pclass = {}'.format(value))
# Adding axes annotations
fig.suptitle('SVM on make_blobs')
plt.show()


