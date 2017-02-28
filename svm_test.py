import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

data = np.loadtxt(r'C:/Users/alexa/Dropbox/TracerCo project team folder/12V motor/reference12V.csv',delimiter=',')
data_anom = np.loadtxt(r'C:/Users/alexa/Dropbox/TracerCo project team folder/12V motor/specific_anomaly_test1.csv',delimiter=',', usecols=range(3))

X, Y = data[:,0][:100], data[:,1][:100]
X_anom, Y_anom = data_anom[:,0][:20], data_anom[:,1][:20]
X_train = np.r_[X + 0.015, X - 0.015]
X_test = np.r_[X_anom + 0.015, X_anom - 0.015]

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_new = clf.predict(X_new)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_new = y_pred_new[y_pred_new == -1].size

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s)
b2 = plt.scatter(X_new[:, 0], X_new[:, 1], c='blueviolet', s=s)
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2],
           ["learned frontier", "training observations",
            "new observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (n_error_train, n_error_new))
plt.show()
