import time as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

# s t a t i s t i k a ###################################################################################################################################
class statistika:
    # uji normalitas
    def normality_test(himpunan):
        """
        masukin himpunan yang mau dites.
        statistika.normality_test(himpunan)
        """
        himpunan = (himpunan - himpunan.mean())/himpunan.std()
        s_normal, p_normal = stats.normaltest(himpunan)
        s_shapiro, p_shapiro = stats.shapiro(himpunan)
        sm.qqplot(himpunan, line='45')
        plt.title('uji normalitas')
        plt.show()
        print(f"normality test p-value: {p_normal:.3f}")
        print(f"shapiro test p-value: {p_shapiro:.3f}")

    # uji kesetaraan varians
    def levene_test(himpunan_pertama, himpunan_kedua, sentral='mean'):
        """
        masukin himpunan pertama.
        masukin himpunan kedua.
        tipe sentral: mean atau median.
        statistika.levene_test(himpunan_pertama, himpunan_kedua, sentral)
        """
        s, p = stats.levene(himpunan_pertama, himpunan_kedua, center=sentral)
        print(f"levene's test p-value: {p:.3f}")

    # uji independent t-test
    def independent_ttest(himpunan_pertama, himpunan_kedua, gaussian=True, levine=True):
        """
        masukin himpunan pertama.
        masukin himpunan kedua.
        data berdistribusi normal (gaussian): True or False.
        data memiliki varians yang sama (levene): True or False.
        statistika.independent_ttest(himpunan_pertama, himpunan_kedua, gaussian, levene)
        """
        if gaussian == True:
            s, p = stats.ttest_ind(a=himpunan_pertama, b=himpunan_kedua, equal_var=levine)
            print(f"independent t-test p-value {p:.3f}")
        else:
            # uji mannwhitney-u
            s, p = stats.mannwhitneyu(x=h1, y=h2)
            print(f"mannwhitney-u test p-value {p:.3f}")

    # uji satu sampel
    def paired_ttest(himpunan_pertama, himpunan_kedua, gaussian=True):
        """
        masukin himpunan pertama.
        masukin himpunan kedua.
        data berdistribusi normal (gaussian): True or False.
        """
        if gaussian == True:
            s, p = stats.ttest_rel(himpunan_pertama, himpunan_kedua)
            print(f"paired t-test p-value: {p:.3f}")
        else:
            s, p = stats.wilcoxon(himpunan_pertama, himpunan_kedua)
            print(f"wilcoxon test p-value: {p:.3f}")
    
    # uji korelasi
    def correl(himpunan_pertama, himpunan_kedua):
        s1, p1 = stats.pearsonr(x=himpunan_pertama, y=himpunan_kedua)
        print(f"pearson test p-value: {p1:.3f} with statistics: {s1:.3f}")
        s2, p12 = stats.spearmanr(a=himpunan_pertama, b=himpunan_kedua)
        print(f"spearman test p-value: {p12:.3f} with statistics: {s2:.3f}")
        s3, p3 = stats.kendalltau(x=himpunan_pertama, y=himpunan_kedua)
        print(f"kendalltau test p-value: {p3:.3f} with statistics: {s3:.3f}")

# m a c h i n e - l e a r n i n g #######################################################################################################################

class machine_learning:
    """
    machine_learning adalah kumpulan fungsi untuk membuat machine learning model.
    x adalah dataset yang berisi fitur untuk memprediksi sesuatu.
    y adalah dataset yang berisi target untuk menjadi tolok ukur
    apakah akurasi machine_learning sudah baik atau masih buruk.
    """
    # inisialisasi
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # deal with categorical data
    def categorical_transform(self):
        """
        categorical_transform digunakan untuk mengubah data yang memiliki kolom kategorikal
        untuk diubah menjadi boolean dan mengembalikan kembali menjadi kategorikal.
        ...
        x_encoder adalah hasil mengubah data dari kolom kategorikal ke boolean.
        variabel x_encoder berbentuk sparse matrix.
        tambahkan .toarray() untuk mengubah menjadi bentuk array.
        ...
        x_decoder adalah hasil mengubah kembali data kolom boolean ke kategorikal.
        """
        self.onehot = OneHotEncoder(drop='first')
        self.x_encoder = self.onehot.fit_transform(self.x).toarray()
        self.x_decoder = self.onehot.inverse_transform(self.x_encoder)

    # split data into train and test
    def train_test(self, categorical=False):
        """
        train_test digunakan membagi dataset menjadi dua bagian:
        train dataset dan test dataset.
        secara default, parameter categorical=False.
        ...
        jika argument True, train_test akan menggunakan dataset
        yang sudah diubah dari kategorikal menjadi boolean.
        jika argument False, train_test akan menggunakan dataset
        murni yang dimasukan.
        ...
        x_train adalah dataset yang berisi fitur yang akan dilatih.
        x_test adalah dataset yang berisi fitur yang akan dites.
        y_train adalah dataset yang berisi target yang akan dilatih.
        y_test adalah dataset yang berisi target yang akan dites.
        """
        if categorical == True:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_encoder, self.y, random_state=42, stratify=y)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, random_state=42, stratify=y)

    # scale dataset
    def scaler(self, scale=None):
        """
        scaler digunakan untuk menyamakan skor tiap fitur yang skalanya berbeda-beda.
        fitur A yang memiliki range 100-1000 dan fitur B yang memiliki range 4-9
        akan mempersulit machine learning untuk mengolah data.
        ada tiga pilihan untuk menggunakan scaler: minmax, standard, dan none.
        ...
        minmax akan mengubah skor semua fitur menjadi skala 0 hingga satu.
        standard akan mengubah skor semua fitur menjadi mean=0, std=1.
        none tidak akan mengubah skor fitur.
        """
        if scale == 'minmax':
            self.scala = MinMaxScaler()
            self.x_train = self.scala.fit_transform(self.x_train)
            self.x_test = self.scala.transform(self.x_test)
        elif scale == 'standard':
            self.scala = StandardScaler()
            self.x_train = self.scala.fit_transform(self.x_train)
            self.x_test = self.scala.transform(self.x_test)
        else:
            self.x_train = self.x_train
            self.x_test = self.x_test         

    def classification(self):
        """
        classification digunakan untuk mencari algoritma terbaik
        dari tujuh algoritma yang sudah dipilih, yaitu:
        classic, decision_tree, random_forest, gradient_boosting,
        svm, knn, dan multi_layer_perceptron.
        outputnya adalah skor akurasi tiap model.
        """
        self.classification_models = [
            LogisticRegression(),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            SVC(),
            KNeighborsClassifier(),
            MLPClassifier()]
        
        self.classification_names = [
            'classic_classification',
            'decision_tree',
            'random_forest',
            'gradient_boosting',
            'support_vector_machine',
            'k-nearest_neighbors',
            'multi_layer_perceptron']

        self.classification_acuracy = []
        self.times = []
        
        for m in self.classification_models:
            self.starting = round(t.time())
            m.fit(self.x_train, self.y_train)
            self.y_pred = m.predict(self.x_test)
            self.score = accuracy_score(y_true=self.y_test, y_pred=self.y_pred)
            self.classification_acuracy.append(self.score)
            self.ending = round(t.time())
            self.time_count = self.ending - self.starting
            self.time_counts = str(self.time_count) + 's'
            self.times.append(self.time_counts)

        self.result = pd.DataFrame({
            "Algorithm":self.classification_names, 
            'Accuracy':self.classification_acuracy, 
            'Time':self.times})
        return self.result.sort_values(['Accuracy'],ascending=False)