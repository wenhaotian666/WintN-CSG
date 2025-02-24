from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


# 加载Olivetti faces数据集
faces = fetch_olivetti_faces()
X = faces.data  # 图像数据
y = faces.target  # 标签

# 显示一些样本
fig, ax = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    ax[i // 5, i % 5].imshow(faces.images[i], cmap='gray')
    ax[i // 5, i % 5].set_title(faces.target[i])
    ax[i // 5, i % 5].axis('off')
plt.show()

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维
n_components = 100  # 保留100个主成分
pca = PCA(n_components=n_components, whiten=True)
X_pca = pca.fit_transform(X_scaled)

# 显示前几个主成分
fig, ax = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    ax[i // 5, i % 5].imshow(pca.components_[i].reshape(faces.images[0].shape), cmap='gray')
    ax[i // 5, i % 5].axis('off')
plt.show()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# 训练SVM分类器
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 打印分类报告
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()