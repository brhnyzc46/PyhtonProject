#fonksiyonlar
from conda.auxlib.type_coercion import numberify
from ruamel.yaml import safe_load_all

# Fonksiyon Okuryazarlığı

print("a", "b")

print("a", "b", sep="__")


## Fonksiyon Tanımlama

def calculate(x):
    print(x * 2)


calculate(5)


# iki argümanlı/parametreleri bir fonksiyon tanımlayalım.

def summer(arg1, arg2):
    print(arg1 + arg2)


summer(7, 8)


#######################################################

# Docstring

def summer(arg1, arg2):
    print(arg1 + arg2)


def summer(arg1, arg2):
    """
    Sum of two numbers
    Args:
        arg1: int, float
        arg2: int, float

    Returns:
        int, float

    """
    print(arg1 + arg2)


summer(1, 3)


##########################################
#Fonksiyonların statement/body bölümü

#def function_name(parameters/arguments):
# statements (function body)


def say_hi(string):
    print(string)
    print("hi")
    print("hello")


say_hi("miuul")


def multiplication(a, b):
    c = a * b
    print(c)


multiplication(10, 9)

# Girilen değerleri bir liste içinde saklayacak fonksyion

list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(1, 8)
add_element(1213, 213)


##################################################

# Ön tanımlı argümanlar parametreler

def divide(a, b):
    print(a / b)


divide(1, 2)


def divide(a, b=1):
    print(a / b)


divide(1)


def say_hi(string="Merhaba"):
    print(string)
    print("hi")
    print("hello")


say_hi("mrb")

##########################################################

# ne zaman fonksiyon yazma ihtiyacımız olur

# varm (ısı), moisture (nem), charge (pil)

(56 + 15) / 80
(17 + 45) / 70


#DRY

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)


calculate(98, 12, 78)


###############################################################

# Return: Fonksiyon çıktılarını girdi olarak kullanma

def calculate(varm, moisture, charge):
    return (varm + moisture) / charge


calculate(98, 12, 78) * 10


def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge

    return varm, moisture, charge, output


calculate(98, 12, 78)

varm, moisture, charge, output = calculate(98, 12, 78)


#########################################################################################

# Fonksiyon içerisinden Fonksiyon Çağırmak

def calculate(varm, moisture, charge):
    return int((varm + moisture) / charge)


calculate(90, 12, 12) * 10


def standardization(a, p):
    return a * 10 / 100 * p * p


standardization(45, 1)


def all_calculation(varm, moisture, charge, p):
    a = calculate(varm,moisture,charge)
    b = standardization(a, p)
    print(b*10)

all_calculation(1,3,5,12)



def all_calculation(varm, moisture, charge, a, p):
    print(calculate(varm,moisture,charge))
    b = standardization(a, p)
    print(b*10)

all_calculation(1,3,5,19 ,12 )

##############################################################

#Lokal Global Değişkenler

list_store = [1,2]

def add_element(a,b):
    c = a * b
    list_store.append(c)
    print(list_store)

add_element(1,9)



##########################################################

# Koşullar (Conditions)

# True - False

1==1
1==2


#if
# eğer true ise ekrana something yazar. Mevzu if in true olma durumudur.

if 1==1:
    print("something")


if 1==2:
    print("something")

number = 10

if number == 10:
    print("number is 10")

number=20

def number_check(number):
    if 1 == 1:
        print("something")

number_check(10)


#################################################################

# else & elif

def number_check(number):
    if number ==10:
        print("number is 10")
    else:
        print("number is not 10")

number_check(12)

# elif    Kontrol etmen gereken durum 2 den fazlaysa elifi kullanabilirsin

def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 110")
    else:
        print("equal to 10")

number_check(1)


#########################################################################

# Döngüler  (loops)

#for loop

students = ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]
students[2]


for student in students:
    print(student)

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

for salary in salaries:
    print(int(salary*20/100 + salary))


for salary in salaries:
    print(int(salary*30/100 + salary))

def new_salary(salary, rate):
    return int(salary*rate/100 + salary)

new_salary(1500, 10)

for salary in salaries:
    print(new_salary(salary, 10))


for salary in salaries:
    if salary >= 3000:
      print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))


#############################################################################################

################################################
# UYGULAMA - MÜLAKAT SORUSU

#AMAÇ: AŞAĞIDAKİ ŞEKİLDE STRİNG DEĞİŞTİREN FONKSİYON YAZMAK İSTİYORUZ.
# çift index büyük tek index küçük

#before: "hi my name is john and i am learning python"
#after:  "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

range(len("miuul"))
range(0, 5)

for i in range(len("miuul")):
    print(i)

#m = "miuul"
#m[0]


def alternating(string):
    new_string =""
    for string_index in range(len(string)):
        if string_index % 2 == 0:
           new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()
    print(new_string)

alternating("hi my name is john and i am learning python")

####################################################################################################

# break continue while

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        break
    print(salary)


for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

### while

number = 1
while number < 5:
    print(number)
    number +=1

########################################################################

# Enumerate : Otomatik counter/indexer ile for loop
# hem öğrencileri temsil etme hemde öğrencilere karşılık gelen indexleri temsil etme

students = ["John", "Mark", "Venessa", "Mariam"]

for student in students :
    print(student)

for index, student in enumerate(students):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)


######################################################################################

#UYGULAMA

#divide_students fonksiyonu yazınız.
#çift indexte yer alan öğrencileri bir listeye alınız.
#tek indexte yer alan öğrencileri başka bir listeye alınız
#fakat bu iki liste tek bir liste olarak return olsun.

students = ["John", "Mark", "Venessa", "Mariam"]

def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)

    print(groups)
    return groups


st = divide_students(students)
st[0]
st[1]


#########################################################################################

#alternating fonksiyonunu enumerate ile yazma

def alternating_with_enumerate(string):
    new_string=""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)

alternating_with_enumerate("hi my name is john and i am learning python")

#########################################################################################

# Zip

students = ["John", "Mark", "Venessa", "Mariam"]

age = [23,30,45,32]

blocks = ["a","b","c","d"]

list(zip(students,age,blocks))


#################################################################################


# lambda, map,filter, reduce,


def summer(a,b):
    return  a + b


summer(1 , 3 )*9
new_sum = lambda a, b: a+b

new_sum(4 , 5)

# map

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries))

list(map(lambda x: x * 20 / 100 + x, salaries))

#del new_sum

# Filter

list_store = [1,2,3,4,5,6,7,8,9,10]
list(filter(lambda x: x % 2 == 0,list_store))

# reduce

from functools import reduce
list_store = [1,2,3,4]
reduce(lambda a,b : a+b , list_store)


###############################################################################################
# COMPREHENSIONS

# List Comprehension

salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(x):
    return x * 20 / 100 + x


for salary in salaries:
    print(new_salary(salary))


null_list = []


for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else :
        null_list.append(new_salary(salary*2))

[new_salary(salary*2) if salary < 3000 else new_salary(salary) for salary in salaries]

# if tekse en sağda kalır ama if ve else kullanacaksan for en sağda kalır


[salary *2  for salary in salaries]

[salary *2  for salary in salaries if salary < 3000 ]

[salary *2  if salary < 3000 else salary * 0 for salary in salaries ]


[new_salary(salary *2 ) if salary < 3000 else new_salary(salary * 0.2) for salary in salaries ]


students = ["John", "Mark", "Venessa", "Mariam"]

students_no = ["John", "Venessa"]


[student.lower() if student in students_no else student.upper() for student in students]


###############################################################################################

# Dict Comprehension


dictionary = { 'a':1,
               'b':2,
               'c':3,
               'd':4,
}

dictionary.keys()
dictionary.values()
dictionary.items()


{k: v ** 2 for (k, v ) in dictionary.items()}

{k.upper(): v  for (k, v ) in dictionary.items()}


###############################################################################################

# Uygulama - Mülakat Sorusu

# Amaç: çift sayıların karesi alınarak bir sözlüğe eklenmek istenmektedir.
# Key' ler orjinal değerler value lar ise değiştirilmiş değerler olcak

numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0 :
        new_dict[n] = n**2


{n: n**2 for n in numbers if n % 2 == 0}


###############################################################################################


# List & Dict Comprehension Uygulamaları

# Bir veri setindeki değişken isimlerini değiştirmek


# before:
# ['total', 'speeding', 'alcohol', 'not_distracted', 'no_ previous', 'ins_premium', 'ins_losses', 'abbrev']

#after:
#['TOTAL', 'SPEEDING', 'ALCOHOL', 'NOT_DISTRACTED', ' NO_ PREVIOUS', 'INS_PREMIUM', 'INS_LOSSES','ABBREV']

import seaborn as sns
df= sns.load_dataset("car_crashes")
df.columns


for col in df.columns:
    print(col.upper())

A = []

for col in df.columns:
    A.append(col.upper())

df.columns = A


df = sns.load_dataset("car_crashes")

df.columns = [col.upper() for col in df.columns]

#########################################################################

# İsminde "INS" olan değişkenlerin başına FLAG diğerlerine NO_FLAG ekle

# "A" + "B" = AB

[col for col in df.columns if "INS" in col]


["FLAG_" + col for col in df.columns if "INS" in col]


["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]


df.columns = ["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]


#########################################################################

# Amaç key i string, value su aşağıdaki gibi bir liste olan sözlük oluşturmak
#Sadece sayısal değişkenler istiyoruz

# Output:
#{'total': ['mean', 'min', 'max', 'var'],
#'speeding': ['mean', 'min', 'max', 'var'],
#'alcohol': ['mean', 'min', 'max', 'var'],
#'not_distracted': ['mean', 'min', 'max', 'var'],
#'no_ previous': ['mean', 'min', 'max', 'var'],
#'ins_premium': ['mean', 'min', 'max', 'var'],
#'ins_losses': ['mean', 'min', 'max', 'var']}

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

# dtype != "0" object olmayan tipteki değişkenleri seçer

num_cols = [col for col in df.columns if df[col].dtype != "O"]

soz = {}

agg_list = ['mean', 'min', 'max', 'var']

for col in num_cols:
    soz[col] = agg_list

#kısa yol   -----> # col kısmı key : agg_list kısmı value

new_dict = {col: agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(new_dict)


















