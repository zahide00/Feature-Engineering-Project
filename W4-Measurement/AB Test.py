# Görev 1: Veriyi Hazırlama ve Analiz Etme
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
pd.set_option('display.float_format', lambda x: '%.2f' % x)
#


# Adım 1: ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı
# değişkenlere atayınız.

df_control= pd.read_excel(r"C:\Users\arsla\PycharmProjects\pythonProject\zahide\ab_testing_veri\ab_testing.xlsx", sheet_name= "Control Group")


df_test= pd.read_excel(r"C:\Users\arsla\PycharmProjects\pythonProject\zahide\ab_testing_veri\ab_testing.xlsx", sheet_name= "Test Group")

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

df_test.describe().T # ortalama ile nonparametrik değerler olan medyan değeri arasında belirgin bir fark yoktur.
df_control.describe().T # ortalama ile nonparametrik değerler olan medyan değeri arasında belirgin bir fark yoktur.


# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.
#
#
concat_df = pd.concat([df_test,df_control])


#---------------------------------------%%%%%%%%%%%-------------------------------------------------



# Görev 2: A/B Testinin Hipotezinin Tanımlanması
# Adım 1: Hipotezi tanımlayınız.
# H0 : M1 = M2  Teklif verme yöntemi arasında müşterilerin satın alma ortalamasında anlamlı bir farklılık gözlemlenmemiştir.
# H1 : M1!= M2   Teklif verme yöntemi arasında müşterilerin satın alma ortalamasında anlamlı bir farklılık vardır. .

# Adım 2: Kontrol ve test grubu için purchase (kazanç) ortalamalarını analiz ediniz.
df_test["Purchase"].mean()
df_control["Purchase"].mean()


#---------------------------------------%%%%%%%%%%%-------------------------------------------------



# Görev 3: Hipotez Testinin Gerçekleştirilmesi
# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.
# Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz.
# Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen p-value değerlerini yorumlayınız.

#----------Normallik Varyasım Kontrolü---------------
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
Test_stas,p_value = shapiro(df_test["Purchase"])
print("test_stat : %.4f , p_value : %.4f " % (Test_stas,p_value))

ts,p_value = shapiro(df_control["Purchase"])
print("ts : %.4f, p_value : %.4f" % (ts,p_value))

#---->>>>>>hem test hem kontrol grubunda p>0.05 old için HO reddedilemez, yani normal varsayımı sağlanmaktadır.

#--------Varyans Homojenlik Kontrollü----------
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
ts,p_value= levene(df_test["Purchase"],df_control["Purchase"])
print("ts : %.4f, p_value :%.4f" % (ts,p_value))

# ---->>>>>>p>0.05 old için Ho reddedilemez, yani varyanslar homojendir.


# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.

# ---->>>>>> Varsayımlar sağlandığı için t testiparametrik test) yapılır

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma ortalamaları arasında istatistiki
# olarak anlamlı bir fark olup olmadığını yorumlayınız.
#
ts , p_value = ttest_ind(df_test["Purchase"],
                              df_control["Purchase"],
                              equal_var=True)
print("ts : %.4f, p_value : %.4f" % (ts,p_value))

#--->>>>>>>p_value > 0.05 old içn ho reddedilemez. yani iki yöntem arasında satın alma ortalamasında anlamlı bir fark yoktur diyebiliriz.

#---------------------------------------%%%%%%%%%%%-------------------------------------------------

# Görev 4: Sonuçların Analizi
# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

#------>>> Normallik Varsayımı ve Homojen Kontrolü Varsayımı sağlandığı için, t_testi kullanıldı.

# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.


# ---->>yeni teklif verme yönteminin uygulandığı test grubunda avarage binding ile satın alma oranlarının mevcut olan maximum binding satın alma oranı arasında anlamlı bir fark olmadığı için, alternetif yöntemin denenmesi daha kazançlı olmayacaktır.