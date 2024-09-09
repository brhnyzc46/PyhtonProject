from pydoc import importfile
from typing import List, Any

##################################################################

# PYTHON İLE VERİ ANALİZİ

# Numpy
# Pandas
# Veri Görselleştirme : Matplotlib & Seaborn
# Gelişmiş Fonksiyon Keşifçi Veri Analizi (Advanced Functional Exploratory Data Analysis)

# Neden Numpy ????
# 1- verimli veri saklar hızlıdır. sabit tipte veri saklar bu yüzden hızlıdır
# 2- fonksiyonel ve vektörel düzeyde kolaylık sağlar.

##################################################################
# PYTHON İLE VERİ ANALİZİ
# Numpy

import numpy as np
from fontTools.unicodedata import block
from scipy.special import linestyle

a=[1,2,3,4]
b=[2,3,4,5]

ab=[]

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

#numpy ile

a = np.array([1,2,3,4])
b=  np.array([2,3,4,5])
a * b

#####################################################################

import numpy as np

np.array([1,2,3,4,5])
type(np.array([1,2,3,4,5]))

np.zeros(10, dtype=int)
np.random.randint(0,10, size=10)
np.random.normal(10, 4, (3,4))

#####################################################################

import numpy as np

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size = 5)
a.ndim
a.shape
a.size
a.dtype

#####################################################################
# reshape

import numpy as np

np.random.randint(1,10, size=9)
np.random.randint(1,10, size=9).reshape(3, 3)

ar = np.random.randint(1,10, size=9)

ar.reshape(3, 3)

#####################################################################

# Index işlemleri

import numpy as np

a = np.random.randint(10, size = 10)
a[0]
a[0:5]
a[0] = 999


m = np.random.randint(10, size = (3,5))


m[0, 0]                       # m[0, 1]  0: satır  //  1: sütun
m[2, 3]
m[2, 3] = 999

m[0,0] = 2.1 # float çalışmaz numpyda

m[: ,0]  # bütün satırları seç ":" demek.tir / ve 0. sütunu seç.

m[0:2, 0:3]

#####################################################################

# Fancy Index
import numpy as np

v = np.arange(0,30,3)  # 0 dan 30 a kadar 3 er 3 er artsın

v[1]

catch = [1,2,3]

v[catch]


#####################################################################

# Numpy'da koşullu işlemler

import numpy as np
v = np.array([1,2,3,4,5])

# Klasik döngüyle

ab= []

for i in v :
    if i < 3:
        ab.append(i)

# Numpy ile
v < 3

v[v < 3]
v[v != 3]


######################################################################################

# Matematiksel İşlemler

import numpy as np

v = np.array([1,2,3,4,5])

v / 5

v * 5 / 10

np.subtract(v,1)
np.add(v,1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)

v = np.subtract(v,1)  # kalıcı atama

# İki bilinmeyenli denklem çözümü


a = np.array([[5,1],[1,3]])
b = np.array([12,10])

np.linalg.solve(a,b)

##################################################################################################################
# PYTHON İLE VERİ ANALİZİ
# PANDAS

import pandas as pd

s = pd.Series([10,77,12,4,5])
type(s)
s.index
s.dtype
s.size
s.ndim  #boyut bilgisi
s.values  #içindeki değerler
type(s.values)
s.head(3)  #ilk 3 değer
s.tail(3)  #son 3 değer

##################################################################################################################

import pandas  as pd
# df = pd.read_csv("dosyanın kayıtlı olduğu yer")
# df.head() verinin çıktısı


################################################################

# Veriye Hızlı Bakış

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()  # ilk n değer
df.tail()  # son n değer
df.shape # boyut bilgisi
df.info()   # değişkenlerin tipi , boş olup olmadığı ve isimleri hakkında bilgi verir.
df.columns  #değişken isimleri
df.index
df.describe().T  # özet istatistiki bilgileri
df.isnull().values.any()  # eksik değer var mı ?
df.isnull().sum()    # her bir değişkende kaç tane eksik değer var

df["sex"].head()
df["sex"].value_counts()

################################################################

# Pandas ta Seçim İşlemleri

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()

df[0:13]
df.drop(0, axis=0).head()


# Değişikliği kalıcı olarak yapmak için df.drop(delete_indexes, axis=0, inplace = True)
delete_indexes = [1,3,5,7]
df.drop(delete_indexes, axis=0).head(10)


# Değişkeni Indekse Çevirmek

df["age"].head()
df.age.head()

df.index = df["age"]
df.drop("age", axis=1).head()

df.drop("age", axis=1, inplace=True)
df.head()

# Indexi Değişkene Çevirmek

df.index

df["age"] = df.index
df.head()
df.drop("age", axis=1, inplace=True)

# 2.yol reset_index fonksiyonu

df.reset_index().head()
df = df.reset_index()
df.head()


################################################################################################################################

# Değişkenler Üzerinde İşlemler (Sütun)

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()
df.age.head()

df["age"].head()
type(df["age"].head())

# seçim sonucunun dataframe olarak kalmasını istiyosan df[["age"]]

df[["age"]].head()
type(df[["age"]].head())

df[["age", "alive"]]

col_names= ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"] **2
df["age3"] = df["age"] / df["age2"]

df.drop("age3", axis=1).head()

df.drop(col_names, axis=1).head()


# df.loc seçme işlemi /  "~" dışındakileri seç
df.loc[:, ~df.columns.str.contains("age")].head()

################################################################################################################################

# loc & iloc
# iloc : int based selection.
# loc : label based selection

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# iloc  [satır,sütun]
df.iloc[0:3]
df.iloc[0, 0]

# loc
df.loc[0:3]
df.loc[0:3, "alive"]

#Birden fazla değişkeni isimlerini ifade ederek seçebilme

col_names = ["age", "embarked","sex"]
df.loc[0:2, col_names]

################################################################################################################################

# Koşullu Seçim

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()

df.loc[df["age"] > 50, ["age","class"]].head()

# eğer 1 den fazla koşul varsa paranteze al


df_new = df.loc[(df["age"] > 50) & (df["sex"] == "male") & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")), ["age","class","embark_town"]]
df_new["embark_town"].value_counts()

################################################################################################################################

# Toplulaştırma ve Gruplama

#count() first() last() mean() median() ......

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()

df.groupby("sex")["age"].mean()  # cinsiyete göre yaş ortalaması

df.groupby("sex").agg({"age":"mean"})
df.groupby("sex").agg({"age":["mean", "sum"]})  # veriyi cinsiyete göre kırıp ortalama sum aldık

df.groupby("sex").agg({"age":["mean", "sum"], "survived": "mean"})

df.groupby(["sex","embark_town","class"]).agg({"age":["mean", "sum"], "survived": "mean"})

df.groupby(["sex","embark_town","class"]).agg({
    "age":["mean"],
    "survived": "mean",
"sex": "count"})

################################################################################################################################

# Pivot Table

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embarked")

df.pivot_table("survived", "sex", "embarked", aggfunc="std")

df.pivot_table("survived", "sex", ["embarked", "class"])

# yaş değişkeni sayısal olduğu için alttaki fonksiyosnu yazıyoruz//// sayısalı kategoriğe çevirmekı için cut fonksiyonu kullan
# qcut elimdeki sayısal değişkenleri bilmiyorum yüzdelik çeyrek olarak bölünsün

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])

df.pivot_table("survived", "sex", ["new_age", "class"])

# karakter sayısını arttırma

pd.set_option('display.width',500)

################################################################################################################################

# Apply & Lambda

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

for col in df.columns:
    if "age" in col:
        df[col] = df[col] /10

df.head()

df[["age","age2","age3"]].apply(lambda x : x /10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x : x /10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x : (x - x.mean()) / x.std()).head()

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

# kaydetmek için df.loc[:, ["age", "age2","age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.loc[:,  df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)
df.head()  # bu da kayıt için



#apply fonksiyonu elimizdeki fonksiyonu satır ve sütunlara uygulamamızı sağlar


################################################################################################################################

# Birleştirme (Join)İşlemleri

import numpy as np
import pandas as pd

m= np.random.randint(1,30, size=(5,3))
df1 = pd.DataFrame(m, columns=["var1","var2","var3"])
df2 = df1 + 99


pd.concat([df1,df2])

pd.concat([df1,df2], ignore_index=True)  # indexler 0' dan başlaması için


# Merge ile Birleştirme

df1 = pd.DataFrame({'employees': ['john','dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering','hr']})

df2 = pd.DataFrame({'employees': ['mark','john', 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})


pd.merge(df1, df2)
pd.merge(df1, df2, on="employees")

# Amaç Her çalışanın müdür bilgisine erişmek istiyoruz

df3 = pd.merge(df1, df2)


df4 = pd.DataFrame({'group': ['accounting', 'engineering','hr'],
                    'manager': ['Burhan', 'Ahmet', 'Banu']})

pd.merge(df3, df4)


################################################################################################################################

# VERİ GÖRSELLEŞTİRME : MATPLOTLIB & SEABORN

# MATPLOTLIB

# Kategorik Değişken : Sütun Grafiği >> countplot  or pasta grafiği >>> barplot
# Sayısal D. : hist, boxplot

##########################################################
# Kategorik Değişken Görselleştirme

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind='bar')
plt.show(block=True)

################################################################################################################################

# Sayısal Değişken Görselleştirme

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show(block=True)

plt.boxplot(df["fare"])
plt.show(block=True)

#########################################################

# matplotlib özellikleri

# plot özelliği
import numpy as np

x = np.array([1,8])
y = np.array([0,150])

plt.plot(x, y)
plt.show(block=True)

plt.plot(x,y, 'o')
plt.show(block=True)

# marker  (işaretleyici) özelliği

y= np.array([13,27,89,12])

plt.plot(y, marker= 'o')
plt.show(block=True)

# line
# dashed : kesikli
# dotted : noktalı
# dashdot: hem nokta hem kesikli

y= np.array([13,27,89,12])

plt.plot(y, linestyle="dashed", color= "r")
plt.show(block=True)

# Multiple Lines
import numpy as np
import matplotlib.pyplot as plt

y= np.array([13,27,89,12])
x= np.array([122,31,23,65])

plt.plot(x)
plt.plot(y)
plt.show(block=True)


# Labels

y= np.array([13,27,89,12])
x= np.array([122,31,23,65])

plt.plot(x,y)
plt.show(block=True)
# Başlık Title
plt. title ("bu ana başlık")

# x eksenini isimlendirme
plt.xlabel("X eksenini isimlendirmesi")

plt.ylabel("Y ekseni isimlendirmesi")

# plt.grid() arkaya ızgara ekler

#Subplots
x = np.array([80,86,90,95])
y = np.array([240,250,260,270])
plt.subplot(1,2,1) # 1 satır 2 sütunlu grafik oluşturcam şuanda 1. grafiğini oluşturdum
plt.title("1")
plt.plot(x,y)
plt.show(block=True)



x = np.array([50,56,60,66])
y = np.array([240,250,260,270])
plt.subplot(1,2,2) # 1 satır 2 sütunlu grafik oluşturcam şuanda 1. grafiğini oluşturdum
plt.title("2")
plt.plot(x,y)
plt.show(block=True)

##################################################################################################################
 # SEABORN

import pandas as pd
import seaborn as sns
from  matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x = df["sex"], data = df)
plt.show(block=True)

df["sex"].value_counts().plot(kind='bar')
plt.show(block=True)

# Sayısal değişkenleri görselleştirme

sns.boxplot(x=df["total_bill"])
plt.show(block=True)

df["total_bill"].hist()
plt.show(block=True)

##################################################################################################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ

# 1. GENEL RESİM

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

def check_df(dataframe, head=5):
    print("########## SHAPE ##############")
    print(dataframe.shape)
    print("####################### TYPES ######################")
    print(dataframe.dtypes)
    print("############# HEAD ##############")
    print(dataframe.head(head))
    print("############ TAIL###############")
    print(dataframe.tail(head))

check_df(df)

df = sns.load_dataset("flights")
check_df(df)


#######################################################################################################

# 2. Kategorik Değişken Analizi 1. Kısım

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"] ]

num_but_cat = [col for col in df.columns if df[col].nunique() < 5 and df[col].dtypes in ['int' , 'float'] ]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str (df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat  # bütün object sinsirella lar cat colsun içinde

cat_cols = [col for col in cat_cols if col not in cat_but_car ]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]

df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##################")

# key var value var
cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)

##########################################################################################
# Kategorik değişken 2.

# Hata var sor

def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plot.show(block=True)



for col in cat_cols:
    if df[col].dtypes == "bool":
        print("asdasdasdasd")
    else :
        cat_summary(df, col, plot=True)

cat_summary(df, "sex", plot= True)




##########################################################################################

# SAYISAL DEĞİŞKEN ANALİZİ

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"] ]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ['int' , 'float'] ]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str (df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat  # bütün object sinsirella lar cat colsun içinde
cat_cols = [col for col in cat_cols if col not in cat_but_car ]


df[["age","fare"]].describe().T  #istatistiksel özellikleri describe

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df, "age", plot=True)


for col in num_cols:
    num_summary(df, col, plot=True)

##########################################################################################

# DEĞİŞKENLERİN YAKALANMASI VE İŞLEMLERİN GENELLEŞTİRİLİMESİ

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

# help(grab_col_names) dersen fonksiyon bilgileri gelir.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik, ve kategorik fakat kardinal değişkenlerin isimleri

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
            numerik
    car_th

    Returns
    -------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken list.
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi
    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ['int', 'float']]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat  # bütün object sinsirella lar cat colsun içinde
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df) # raporlama için

#####################################################################################
# HEDEF DEĞİŞKEN ANALİZİ ÖNEMLİ!!!!!!




####################################################################################
#KORELASYON ANALİZİ

# df = df.iloc[:, 1:-1]  1 den -1 e kadar git , id ve unnamed değişkenlerinden kurtul

# corr = df[num_cols].corr()  --> num_cols'lara göre korelasyon alır

# Korelasyonları ısı haritasında gösterme
# sns.set(rc={'figure.figsize':(12,12)})
# sns.heatmap(corr, cmap= "RdBu")
# plt.show()















