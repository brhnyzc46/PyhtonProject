#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.
file_path = "ab_testing.xlsx"
control_group = pd.read_excel(file_path, sheet_name="Control Group")
test_group = pd.read_excel(file_path, sheet_name="Test Group")

print(control_group.head())
print(test_group.head())

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

print(control_group.describe().T)
print(control_group.shape)
print(control_group.info())
print(test_group.describe().T)
# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

birlestirme= pd.concat([control_group,test_group], axis=0)
birlestirme.head()


#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.
# H0: M1 = M2
# kontrol ve test grubu ortalamaları arasında fark yoktur.

# H1: M1 != M2
# kontrol ve test grubu ortalamaları arasında fark vardır.


# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz.
control_group["Purchase"].mean()
test_group["Purchase"].mean()
#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

test_stat, pvalue = shapiro(control_group["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat,pvalue = shapiro(test_group["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# H0 reddedilemez normallik varsayımı sağlanmıştır.


test_stat, pvalue = levene(control_group["Purchase"], test_group["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# H0 Reddedilemez varyanslar homojendir.





# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz
test_stat, pvalue = ttest_ind(control_group["Purchase"], test_group["Purchase"], equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

# p value 0.3493 çıktığından dolayı H0 reddedilemez ve kontrol ve test grubu satın alma ortalamaları arasında ist. olarak anlamlı bir fark yoktur.


##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

# Bağımısz iki örneklem T testini kullandım. İlk olarak normallik testi uyguladığımda iki grubun normal dağıldığını gördüm.
# Ardından varyansların homojenliği testini uyguladığımda varyanslarında homojen olduğunu gördüm.
# Bundan dolayı Bağımsız İki Örneklem T testi uyguladım.



# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.
"""
Bu p-değerine göre (0.349 > 0.05), kontrol ve test grubu arasında satın alma ortalamaları açısından istatistiksel olarak anlamlı bir fark yoktur.
 Yani, "average bidding" ile "maximum bidding" arasında satın alma açısından belirgin bir fark olmadığı sonucuna varıyoruz.
Sonuç olarak, bu A/B testi sonuçlarına dayanarak müşteriye şu tavsiyede bulunabiliriz: 
Yeni teklif stratejisi olan "average bidding" ile mevcut "maximum bidding" stratejisi arasında satın almalar açısından önemli bir fark bulunmamaktadır.
 Bu durumda mevcut stratejiyi sürdürmek ya da yeni stratejiyi başka metriklerle test etmek mantıklı olabilir.
"""
