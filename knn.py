import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Muat dataset
file_path = "Social_Network_Ads2.csv"
dataset = pd.read_csv(file_path)

# Mendefinisikan fitur (X) dan target (y)
X = dataset[['Age', 'EstimatedSalary']].values
y = dataset['Purchased'].values

# Membagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Melakukan normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Membangun model K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred = knn_model.predict(X_test)

# Confusion Matrix dan Akurasi
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Akurasi:", accuracy * 100, "%")

# Visualisasi data latih
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
                     np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01))

# Kontur dan sebaran data latih
plt.figure(figsize=(10, 6))
plt.contourf(X1, X2, knn_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.3, cmap=ListedColormap(('lightcoral', 'palegreen')))

# Menampilkan data titik untuk setiap kelas
color_map = ListedColormap(['crimson', 'forestgreen'])
for idx, label in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == label, 0], X_set[y_set == label, 1],
                color=color_map(idx), edgecolor='black', s=50, label=f'Kelas {label}')

# Menambahkan label sumbu, judul, dan legenda
plt.title('K-Nearest Neighbors (Data Latih)')
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.legend()
plt.show()
