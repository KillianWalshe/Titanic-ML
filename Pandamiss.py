import pandas as pd
titanic_survival= pd.read_csv('Datafiles/titanic_survival.csv')

age_null=pd.isnull(titanic_survival["age"])
age_null_true=titanic_survival[age_null]
age_null_count=len(age_null_true)
print(age_null_count)



#ages=titanic_survival["age"]
#age=0
#for i in range(len(titanic_survival["age"])):
#    if age_null[i]==False:
#        #print(i)
#        age+=ages[i]


titanic_age=titanic_survival["age"][age_null==False]

correct_mean_age=sum(titanic_age)/len(titanic_age)
print(correct_mean_age)  

correct_mean_fare = titanic_survival["fare"].mean()         
print(correct_mean_fare)  

#passenger_classes = titanic_survival.pclass.unique()
fares_by_class={}
passenger_classes=[1,2,3]
pclass_list=titanic_survival["pclass"]
for i in passenger_classes:
    print(i)
    true_rows=pclass_list==i
    fare_rows=titanic_survival["fare"][true_rows==True]
    fare_mean=fare_rows.mean()
    fares_by_class[i]=fare_mean
print(fares_by_class)



import numpy as np

#does same as above
passenger_class_fares = titanic_survival.pivot_table(index="pclass", values="fare", aggfunc=np.mean)
#print(passenger_class_fares)

passenger_age = titanic_survival.pivot_table(index="pclass", values="age", aggfunc=np.mean)
print(passenger_age)

fare_survived=["fare","survived"]
port_stats=titanic_survival.pivot_table(index="embarked", values=fare_survived, aggfunc=np.sum)

print(port_stats)
import math
ports=titanic_survival["embarked"].unique()
ports=([i for i in ports if str(i)!='nan'])
print(ports)


description=""
titanic_survival["description"]=description
ports_des=["Southampton","Cork","Queens"]
for value in range(len(ports)):
    true_rows=titanic_survival["embarked"]==ports[value]
    titanic_survival.loc[true_rows,"description"]=ports_des[value]

print(titanic_survival["embarked"][0:10])
print(titanic_survival["description"][0:10])   



drop_na_columns= titanic_survival.dropna(axis=1)
drop_na_rows= titanic_survival.dropna(axis=0)
new_titanic_survival= titanic_survival.dropna(axis=0,subset=["age","sex"])

first_ten_rows=new_titanic_survival.loc[0:5]
#print(first_ten_rows)


#reset index can convert back to dataframe
titanic_reindexed=new_titanic_survival.reset_index(drop=True)
#print(titanic_reindexed.iloc[0:5,0:3])

def null_count(column):
    # Extract the hundredth item
    age_null=pd.isnull(column)
    age_null_true=column[age_null]   
    age_null_count=len(age_null_true)
    return age_null_count
column_null_count = titanic_survival.apply(null_count)
print(column_null_count)


def age_group(row):
    age=row["age"]
    if pd.isnull(age):
        return("Unknown")
    elif age<18:
        return("minor")
    else:
        return("adult")
age_labels=titanic_survival.apply(age_group, axis=1)
print(age_labels)

titanic_survival["age_labels"]=age_labels
age_group_survival=titanic_survival.pivot_table(index="age_labels", values="survived", aggfunc=np.mean)
print(age_group_survival)
