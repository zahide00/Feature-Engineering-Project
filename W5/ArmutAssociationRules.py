import pandas as pd
!pip install mlxtend
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
#
# Adım 1: armut_data.csv dosyasını okutunuz.

df_ = pd.read_csv(r"C:\Users\arsla\PycharmProjects\pythonProject\zahide\Recommendation\Projeler - Recommendation\armut_data\armut_data.csv")

df = df_.copy()

df.head()

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir. ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri
# temsil edecek yeni bir değişken oluşturunuz. Elde edilmesi gereken çıktı:
def concat(x,y):
    z = str(x) + str("_") + str(y)
    return z

df["CreateDate"].min()
df["CreateDate"].max()


df["hizmet"] = df.apply( lambda  x: concat( x["ServiceId"], x["CategoryId"] ), axis=1)
#
# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır. Association Rule
# Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir. Burada sepet tanımı her bir müşterinin aylık aldığı
# hizmetlerdir. Örneğin; 25446 id'li müşteri 2017'in 8.ayında aldığı 4_5, 48_5, 6_7, 47_7 hizmetler bir sepeti; 2017'in 9.ayında aldığı 17_5, 14_7
# hizmetler başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren
# yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.
# Elde edilmesi gereken çıktı:

df['New_Date'] = pd.to_datetime(df['CreateDate'],format='%Y-%m').dt.to_period('M')
df.head()

date_2018_01 = [x for x in df.columns.values if "2018" in x]


df[df["CreateDate"].str.contains("2018-08")]


for i in [2017,2018]:
    df["Create"].str.contains(i)
    print (df)



def concat_year_month(x,y):
    str(x) + str(y)

