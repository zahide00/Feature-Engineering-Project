import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
%pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
#
# Adım 1: Genel resmi inceleyiniz.

df.head()

dff = pd.read_csv(r"C:\Users\arsla\PycharmProjects\pythonProject\zahide\Featur Enginnering\Proje Zipleri\diabetes.csv\diabetes.csv")
df = dff.copy()
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat= [col for col in dataframe.columns if dataframe[col].dtypes != "O" and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + cat_but_car
    cat_cols = [col for col in cat_cols if col not in cat_but_car ]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    print(f"cat_cols : {dataframe[cat_cols].values}")
    print(f"num_cols : {dataframe[num_cols].columns.values}")
    print(f"num_but_cat : {dataframe[num_but_cat].columns.values}")
    return cat_cols,num_cols,cat_but_car


# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

grab_col_names(df)


num_cols = [col for col in df.columns if "Outcome" not in col]

df.head(10)

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)


#kategorik değişken yok
#hedef değişkene göre nümerik değişkenlerin ataması
df.groupby("Outcome")["Pregnancies"].mean()

# df.apply(lambda x: x.mean() if x in x.columns.values) ~~~~~~~????

for x in df.columns.values:
    outcomes = df.groupby("Outcome")[x].mean()
    print(outcomes)

#
#
# Adım 5: Aykırı gözlem analizi yapınız.

# sns.boxplot(x=df["Outcome"])      ~~~~~~~????
# plt.show()


#Eşik değer belirleme
def outliers_outcome(dataframe, col_name, q1 =0.25, q3=0.75):
    quantile1= dataframe[col_name].quantile(q1)
    quantile3= dataframe[col_name].quantile(q3)
    iqr = quantile3-quantile1
    low_limit =  quantile1 - 1.5*iqr
    up_limit = quantile1 + 1.5*iqr
    return up_limit,low_limit
#aykırı değer var mı?


up,low= outliers_outcome(df, "Insulin")

df["Insulin"].describe()

df[(df["Insulin"]<low) | (df["Insulin"] > up)]


def check_outliers(dataframe, col_name):
    up,low = outliers_outcome(dataframe,col_name)
    if dataframe[(dataframe[col_name]< low) | (dataframe[col_name] > up)].any(axis=None):
        return True
    else:
        return False

# df.apply(lambda  x : outliers_outcome(x,x[])) ~~~~~~~~?


check_outliers(df,"Insulin")


def replace_with_thresholds(dataframe, col_name):
    up_limit,low_limit = check_outliers(dataframe,col_name)
    df.loc[df[col_name] < low_limit , col_name] = low_limit
    df.loc[df[col_name] > up_limit , col_name] =up_limit
    return dataframe


check_outliers(df,"Insulin")


def remove_outliers(dataframe, col_name):
    up_lim, low_lim = outliers_outcome(dataframe,col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] >up_lim) | (dataframe[col_name] < low_lim))]
    return  df_without_outliers

for col in num_cols:
     df = remove_outliers(df,col)


for col in num_cols:
    new_df = remove_outliers(df, col)

check_outliers(df,"DiabetesPedigreeFunction")
outliers_outcome(df,"DiabetesPedigreeFunction")
remove_outliers(df,"DiabetesPedigreeFunction")
df.head()
new_df.head()
# Adım 6: Eksik gözlem analizi yapınız.
df.isnull().values.any()

df.isnull().sum()

df[df.isnull().any(axis=1)]

df.isnull().sum().sum()




# Adım 7: Korelasyon analizi yapınız.



# Görev 2 :

#
# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.

def missing_values(dataframe,col_names):
     outcome = dataframe[col_names].apply(lambda x: "NaN" if (x == 0) else x)
     return outcome

df["Insulin"].apply(lambda  x : "NaN" if (x == 0) else x )

missing_values(df,"Insulin")


# for col in num_cols:
#     print(col, check_outlier(df, col))
#
for col in num_cols:
    replace_with_thresholds(df, col)
#
# for col in num_cols:
#     print(col, check_outlier(df, col))
#
# na_names = [col.values== for col in df.columns if (df[col].values.any(axis=0) == "0")] ~~~~???

df.head()
# Adım 2:  Yeni değişkenler oluşturunuz.

# df["Have_Pregnancy"]= df.apply( lambda x : 1 if x["Pregnancies"].value_counts() > 0 else 0 )~~~~????
df.head()

df.loc[df["Pregnancies"].values> 0, "Have_Pregnancy"] = 1
df.loc[df["Pregnancies"].values == 0, "Have_Pregnancy"] = 0
df.head(10)

df["Age_qcut"] = pd.qcut(df["Age"], 5 , labels=["1","2" ,"3" ,"4","5"])

outliers_outcome(df,"DiabetesPedigreeFunction")
check_outliers(df, "DiabetesPedigreeFunction")

#"DiabetesPedigreeFunction" değğişkende 1 den büyük olan değerler atılır.
df.drop(df.loc[df["DiabetesPedigreeFunction"] > 1].index, inplace=True)

df.loc[df["DiabetesPedigreeFunction"] > 1]

df.groupby("Have_Pregnancy")["Outcome"].mean()

df = df.loc[df["DiabetesPedigreeFunction"] > 1, "DiabetesPedigreeFunction"]
df.head()
df.drop("Age_qcut", axis = 1, inplace = True)
# Adım 3: Encoding işlemlerini gerçekleştiriniz

# le= LabelEncoder()
# le.fit_transform(df[col_name]) ~~~~~~~değişkenler sayısal ?

grab_col_names(df)
df.info()
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

num_cols = [col for col in df.columns if col not in ["Outcome"] ]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()

#geri almak
df[num_cols] = scaler.inverse_transform(df[num_cols])

# Adım 5: Model oluşturunuz.

y=df["Outcome"]
x = df.drop(["Outcome"],axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(x_train,y_train)
y_pred = rf_model.predict(x_test)
accuracy_score(y_pred,y_test)


# accuracy_score = %86 olarak çıktı.