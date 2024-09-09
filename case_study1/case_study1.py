from statistics import median

from envs.my_env.DLLs.unicodedata import category
from numpy.ma.core import append
from pandas import unique
from pandas.core.algorithms import nunique_ints
from pandas.core.computation.expr import intersection

x=8
z= 8j + 18
a= True
b= 23 < 22
c= {"Name": "Jake",
    "Age": 27,
    "adress:": "downtowjn"}

s= {"python","mac.le", "data science"}
type(s)

##########################################

text = ("The goal is to turn data into information  and information into insight ")
text_1 = str(text).upper().split()

###############################################################

lst = ["D","A","T","A","S","C","I","E","N","C","E"]
len(lst)
lst[0]
lst[10]
lst[0: 4]
del lst[8]
lst.append("N")
lst.insert(8, "N")
del lst[11]

########################################

dict= {'Christan': ["America",18],
       'Daisy': ["England",12],
       'Antonio': ["Spain",22],
       'Dante': ["Italy",25]}

dict.keys()
dict.values()
dict['Daisy'] = ["England", 13]
dict['Ahmet'] = ["Turkey",24]
dict.__delitem__('Antonio')

########################################
# Görev 5
l = [2, 13, 18, 93, 22]

def func(sayilar):
    tek_sayilar = []
    cift_sayilar = []

    for sayi in sayilar:
        if sayi % 2 == 0 :
            cift_sayilar.append(sayi)
        else :
            tek_sayilar.append(sayi)
    return tek_sayilar, cift_sayilar

tekler, ciftler = func(l)

print("Tek sayılar:", tekler )
print("Çift Sayılar:" ,ciftler)

################################################################################
#Görev 6

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

for index, ogrenci in enumerate(ogrenciler, 1):
    print(index, ogrenci)

print("Mühendislik Fakültesi:")
for index, ogrenci in enumerate(ogrenciler[:3],1):
    print(f"{index}. {ogrenci}")

print("Tıp Fakültesi:")
for index, ogrenci in enumerate(ogrenciler[3:],1):
    print(f"{index}. {ogrenci}")

################################################################################
# Görev 7  Çıktıyı Sor

ders_kodu = ["IST2001","IST2002","IST2003"]
kredi = [3,4,3]
kontenjan = [30,35,40]


list(zip(ders_kodu,kredi,kontenjan))

################################################################################

# Görev 8

kume1 = set(["data","python"])
kume2 = set(["data","function","qcut","lambda","python","miuul"])


kume1.issuperset(kume2)

if kume1.issuperset(kume2):
    print("1. küme 2. kümeyi kapsar")
else:
    fark = kume2.difference(kume1)
    print(fark)

#######################################################################################

# CASE STUDY 2
# # GÖREV 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.


import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns


df.columns = ["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]

print(df.columns)

# df[col].dtype != "0"  numerik demek
#######################################################################################

# # GÖREV 2: List Comprehension yapısı kullanarak car_crashes verisindeki isminde "no" barındırmayan değişkenlerin isimlerininin sonuna "FLAG" yazınız.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

df.columns = [col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]
print(df.columns)

#######################################################################################
# # Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
df.head()

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]

new_df = df[new_cols]
print(new_df.head())

#######################################################################################
# CASE STUDY 3


import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = sns.load_dataset("Titanic")

df.head()
df["sex"].value_counts()  # cinsiyet değişkenininden kaçar tane var

df.nunique()   #her bir sütuna ait unique değerleri

df["pclass"].nunique()  # pclass değişkenine göre unique sayısı

uniques = {
'pclass': df["pclass"].nunique(),
'parch': df["parch"].nunique()
}  ###

print(df["embarked"].dtype)
df["embarked"] = df['embarked'].astype('category')   ##

embarked_C = df[df['embarked'] == 'C']
print(embarked_C)  ###

embarked_not_S = df[df['embarked'] != 'S']
print(embarked_not_S)

passangers = df[(df["sex"] == "female") & (df["age"] < 30)]
print(passangers)

passangers1 = df[(df["fare"] >= 500) | (df["age"] > 70)]
print(passangers1)

df.isnull().sum()

df.drop(columns=["who"])

mode_deck = df["deck"].mode()[0]
df['deck'] = df['deck'].fillna(mode_deck)   ##
print(df['deck'].head())
print(df['deck'].isnull().sum())

median_age = df['age'].median()
df['age'] = df['deck'].fillna(median_age)
print(df['age'].head())
print(df['age'].isnull().sum())

kirilma = df.groupby(['pclass','sex'])['survived'].agg(['sum','count','mean']) ##
print(kirilma)










