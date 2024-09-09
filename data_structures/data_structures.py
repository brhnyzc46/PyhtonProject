

#veri yapılarına giriş ve özet

#sayilar : integer

x = 46
type(x)

#float
x= 10.3
type(x)

# complex
x = 2j + 1
type(x)

#string
x = "hello ai era"
type(x)

#boolean
True
False
type(True)
5==4
type(3==2)

#listeler

x = ["btc","eth"]
type(x)

#sözlük dictionary
x = {"name":"peter", "age":36}
type(x)

#tuple (demet)

x = ("python","sas")
type(x)

#set
x = {"asd","sas"}
type(x)

#not: liste, tuple set ve dict veri yapıları aynı zamanda Python Collections(Arrays) olarak geçer



#############################################################


#sayılar : int ,float ,complex

a = 5
b = 10.5

a * 3
a / 7
a * b/ 10
a ** 2


# tipleri değiştirmek

int(b)
float(a)

int(a * b / 10)   #önce parantez içi çalışır

c = a * b / 10

int(c)



# strings karakter dizileri

print("add")  #ekrana bilgi paylaşırken print kullan
"add"


name = "ocak"


#çok satırlı karakter dizileri

"""veri yapıları cart curt"""
long_str = """asdasdasdasdasdasdasd"""


#karakter dizilerinin elemanlarına erişmek
name
name[0]
name[3]
name[2]

# karakter dizilerinde slice işlemi

name[0:2]  #2 ye kadar git 2 hariç

long_str[0:9]

# string içerisinde karakter sorgulama

long_str

"asd" in long_str



#########################################################

#string metodları

dir(str)


#len

name = "buran"
type(name)
type(len)

len(name)
len("naber")
#bir fonksiyon class yapısının içindeyse buna method denir.
#eğer class yapısı içinde değilse fonksiyondur.


#upper() lower() dönüşümleri

"buran".upper()
"buran".lower()

#replace karakter değiştirme

hi = "hello ai era"
hi.replace("l","p")

#split böler
"Hello AI Era".split()

#strip kırpar
"  ofofo ".strip()
"ofofo".strip("o")

#capitalize ilk harfi büyütür

"foo".capitalize()

dir("foo")

"foo".startswith("f")

##################################


#Liste (List)
#-Sıralıdır. Index İşlemleri yapılabilir ve Kapsayıcıdır.

notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b", "v", "d"]

not_nam = [1, 2, 3, "a", "b", True, [1,2,3]]

not_nam[0]
not_nam[5]
not_nam[6]
not_nam[6][1]

type(not_nam[6])
type(not_nam[6][2])

notes[0] = 99

not_nam[0:4]


#Liste metodları

dir(notes)

len(notes)
len(not_nam)

#listeye eleman ekleme append

notes
notes.append(100)

#pop indekse göre eleman siler

notes.pop(0)

#insert indekse ekler

notes.insert(2, 99)


################################################################


# Sözlük (Dictionary)

# Değiştirilebilir   kapsayıcı ve sıralı ve sırasızdır



# key-value

dictionary = {"REG": "Regression","LOG": "Logistic Regression","CART": "Classification and Reg"}

dictionary["REG"]


dictionary = {"REG": ["RMSE", 10], "LOG": ["MSE", 20],"CART": ["SSE", 30]}

dictionary["CART"][1]

# Key sorgulama

"REG" in dictionary

# Key e göre value ye erişmek
dictionary["REG"]
dictionary.get("REG")


# Value Değiştirmek

dictionary["REG"] = ["YSA", 10]


#Tüm Key ve Value lara erişmek

dictionary.keys()
dictionary.values()

# Tüm çiftlere Tuple Halinde Listeye Çevirme

dictionary.items()

# Key value değerlerini güncellemek

dictionary.update({"REG": 11})

#Yeni Key- Value Eklemek

dictionary.update({"RF": 10})


################################################################

# Tuple demet
# değiştirilemez sıralıdır kapsayıcıdır


t = ("john", "mark", 1, 2)
type(t)

t[0]
t[0:3]

t[0] = 99

t = list(t)
t[0] = 99
t = tuple(t)


######################################################################

# Set Kümeler

#değiştirilebilir sırasız eşsizdir kapsayıcıdır

#difference(): iki kümenin farkı

set1 = set([1,3,5])
set2 = set([1,2,3])

set1 - set2

#set1 de olup set2 de olmayan nelerdir

set1.difference(set2)

#set2 de olup set1 de olmayanlar

set2.difference(set1)



# symmetric_difference() : iki kümede de birbirlerine göre olmayanları getirir.

set1.symmetric_difference(set2)

# intersection() kesişim

set1 = set([1,3,5])
set2 = set([1,2,3])

set1.intersection(set2)

set1 & set2

# union() İki kümenin birleşimi

set1.union(set2)
set2.union(set1)

# isdisjoint() iki kümenin kesişimi boş mu  ----> is ile başlayanlardan true false cevabı beklenir

set1 = set([1,3,5])
set2 = set([1,2,3])

set1.isdisjoint(set2)

# issubset() bir küme diğer kümenin alt kümesi mi ?

set1 = set([7,8,9])
set2 = set([5,6,7,8,9,10])

set1.issubset(set2)
set2.issubset(set1)

#issuperset bir küme diğerini kapsıyor mu

set2.issuperset(set1)






















