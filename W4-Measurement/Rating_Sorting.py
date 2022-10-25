import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Görev 1: Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.
# Adım 1: Ürünün ortalama puanını hesaplayınız.
df = pd.read_csv(r"C:\Users\arsla\PycharmProjects\pythonProject\zahide\amazon_review_veri\amazon_review.csv")
df.head()
df.columns
df["overall"].mean()
# -------------------%%%%%%%%%%%%%%%%%%%%%%%%--------------
# Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.
df["reviewTime"].max()
current_date = pd.to_datetime("2014-12-09")
type(df["reviewTime"])
df["reviewTime"]= pd.to_datetime(df["reviewTime"])
df["days"] = (current_date - df["reviewTime"]).dt.days
df["days_cut"] = pd.qcut(df["days"] ,4 , labels=["1","2","3","4"])
#1. grup en yakın tarih.
def time_based_rating(dataframe, w1,w2,w3,w4):
    return dataframe.loc[dataframe["days_cut"] == "1","overall"].mean() * w1/100 + \
    dataframe.loc[(dataframe["days_cut"] == "2") ,"overall"].mean() * w2/100+ \
    dataframe.loc[(dataframe["days_cut"] == "3"),"overall"].mean() * w3/100 + \
    dataframe.loc[(dataframe["days_cut"] == "4"), "overall"].mean() * w4/100

time_based_rating(df,40,30,20,10)

df.loc[df["days_cut"]=="1", "overall"].mean()
# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.

for i in range (1,5):
    print (df.loc[df["days_cut"] == str(i), "overall"].mean())



# -------------------%%%%%%%%%%%%%%%%%%%%%%%%--------------

# Görev 2: Ürün için ürün detay sayfasında görüntülenecek review’i belirleyiniz.
# Adım 1: helpful_no değişkenini üretiniz.
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# Adım 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.
df["score_pos_neg_diff"] = df["helpful_yes"] - df["helpful_no"]
df["score_average_rating"] = df["helpful_yes"] /  df["total_vote"]

def score_pos_neg_diff(up, down):
    return up - down

score_pos_neg_diff(df["helpful_yes"],df["helpful_no"])

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(df["helpful_yes"], df["helpful_no"]) ####?

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

wilson_lower_bound(df["helpful_yes"].sum(),df["helpful_no"].sum())

df["wilson_lower_bound"] = df.apply( lambda  x : wilson_lower_bound( x["helpful_yes"] , x["helpful_no"]) ,axis=1 )


# Adım 3: 20 Yorumu belirleyiniz ve sonuçları Yorumlayınız.

df["wilson_lower_bound"].sort_values(ascending=False)

#her bir yorumla etkileşime girecek olan kullanıcıların yoruma verecekleri up oranının alt sınırı,
# %5 hata payı wlb fonksiyonunda çıkacak oran olacaktır.
