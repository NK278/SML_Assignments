# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from collections import defaultdict


class QDA_clf:
  def __init__(self):
    self.estimates = None
  def train(self, X_train, y_train):
        self.estimates = self.compute_estimates(X_train, y_train)
  def compute_estimates(self, X_train, y_train):
    df_ = pd.concat([X_train, y_train], axis=1)
    df_.set_index('Label', inplace=True)
    estimates = []
    mv={}
    cov_v={}
    for l in range(len(df_.index.unique())):
      mv[l]=(df_.loc[df_.index==l]).mean(axis=0)
      cov_v[l]=(df_.loc[df_.index==l]).cov()
    lm=1e-6
    det_={}
    inv_cv={}
    for l, v in cov_v.items():
      v_r=v+lm*np.eye(v.shape[0])
      d=np.linalg.det(v_r)
      if d==0 or np.isinf(d):
        d=np.finfo(float).eps
        inv_cov=np.linalg.pinv(v_r)
      else:
        inv_cov=np.linalg.inv(v_r)
      det_[l]=d
      inv_cv[l]=inv_cov
    unique_classes, class_counts = np. unique(y_train, return_counts=True)
    total_samples = len(y_train)
    class_priors = {}
    for c, count in zip(unique_classes, class_counts):
        prior = count / total_samples
        class_priors[c] = prior
    estimates.append((mv,cov_v,det_,inv_cv,class_priors))
    return estimates

  def predict(self,x_test):
    mv=self.estimates[0][0]
    cov_v=self.estimates[0][1]
    det_=self.estimates[0][2]
    inv_cv=self.estimates[0][3]
    class_priors=self.estimates[0][4]
    y_pca_pred=[]
    for i, r in x_test.iterrows():
      mx_dis=float('-inf')
      pred=None

      for l, v in cov_v.items():
        mn_v=mv[l]
        iv=inv_cv[l]
        prior=class_priors[l]
        quadratic_term = np.dot(np.dot((r - mn_v), iv), (r - mn_v))
        discriminant = -0.5 * np.log(det_[l]) - 0.5 * quadratic_term + np.log(prior)
        if discriminant>mx_dis:
          mx_dis=discriminant
          pred=l
      y_pca_pred.append(pred)
    return y_pca_pred

def image_plot(x,x_img):
  num_samples_per_class = 5
  fig, axes = plt.subplots(10, num_samples_per_class, figsize=(12, 12))
  plt.subplots_adjust(wspace=0.3, hspace=0.3)
  for i in range(10):
      class_samples_indices = class_idx(x)[i][:num_samples_per_class]
      for j, idx in enumerate(class_samples_indices):
          axes[i, j].imshow(x_img[idx], cmap='gray')
          axes[i, j].axis('off')

          if j == 0:
              axes[i, j].set_title(f" Class {i} ")

  plt.show()

def class_idx(x):
  class_indices = {i: [] for i in range(10)}
  for i, label in enumerate(x):
    class_indices[label].append(i)
  return class_indices

mnist_data = mnist.load_data(path="mnist.npz")
(train_images, train_labels), (test_images,test_labels) = mnist_data
train_images=train_images.astype(np.float64)/255
test_images=test_images.astype(np.float64)/255

num_samples, width, height = train_images.shape
num_samples2, width2, height2 = test_images.shape
train_images_vectorized = train_images.reshape(num_samples, width * height)
test_images_vectorized = test_images.reshape(num_samples2, width2 * height2)
print("Original shape of images:", train_images.shape)
print("Vectorized shape of images:", train_images_vectorized.shape)
print("Original shape of images:", test_images.shape)
print("Vectorized shape of test images:", test_images_vectorized.shape)

x_train=pd.DataFrame(train_images_vectorized)
y_train = pd.DataFrame(train_labels, columns=['Label'])
x_test=pd.DataFrame(test_images_vectorized)
y_test=test_labels

df = pd.concat([x_train, y_train], axis=1)
df.set_index(y_train.columns[0], inplace=True)
# Sampling 100 samples from x_train
sampled_df = pd.DataFrame()
for label in df.index.unique():
    sampled_data = df.loc[label].sample(n=100, random_state=42)
    sampled_df = pd.concat([sampled_df, sampled_data])

sampled_df.reset_index(inplace=True)
df_copy=sampled_df.drop(columns=[y_train.columns[0]]).copy()
sampled_df_r=df_copy.values.reshape(1000,28,28)

image_plot(x_img=sampled_df_r,x=sampled_df['Label'])
X=df_copy.T.values
means=np.mean(X,axis=1,keepdims=True)
X_cen=X-means
S=(X_cen@X_cen.T)/999
eigen_val,eigen_vec=np.linalg.eigh(S)
sorted_indices = np.argsort(eigen_val)[::-1]
U = eigen_vec[:, sorted_indices]
Y=U.T@X_cen
X_recon=U@Y
squared_diff = (X_cen - X_recon) ** 2
MSE=np.abs(np.mean(squared_diff))
p_values = [5, 10, 20,100,783]
UpY_dict={}
for p in p_values:
  Up=U[:,:p]
  UpY=Up@(Up.T@X_cen)
  UpY_dict[p]=UpY
Up_X_dict={}
for p in p_values:
  Up=U[:,:p]
  Up_X_dict[p]=Up.T@X_cen
UpY_with_means = {}
for idx, (p, UpY) in enumerate(UpY_dict.items(), start=1):
    UpY_m = UpY + means
    UpY_with_means[p]= UpY_m
UpY_dfs={}
for p,UpY in UpY_with_means.items():
  UpY_dfs[p]=UpY_with_means[p].T.reshape(1000,28,28)

x_train_pca_dfs={}
for p, UpX in Up_X_dict.items():
  x_train_pca_dfs[p]=pd.DataFrame(UpX.T)
y_train_pca=pd.DataFrame(sampled_df['Label'],columns=['Label'])
for p in p_values:
  print(f'Image for p={p}')
  image_plot(x_img=UpY_dfs[p],x=sampled_df['Label'])

df_e = pd.concat([x_test, pd.DataFrame(y_test,columns=['Labels'])], axis=1)
sampled_df_t = pd.DataFrame()
for label in df_e['Labels'].unique():
    sampled_data = df_e[df_e['Labels'] == label].sample(n=100, random_state=42)
    sampled_df_t = pd.concat([sampled_df_t, sampled_data])

sampled_df_t.reset_index(inplace=True, drop=True)
y_test_l=sampled_df_t['Labels']
samples_per_class = 100

sample_matrices = []

for label in range(10):

    class_samples = x_test[y_test == label][:samples_per_class]

    sample_matrices.append(class_samples.values.T)


X_test_matrix = np.concatenate(sample_matrices, axis=1)
means_test=np.mean(X_test_matrix,axis=1,keepdims=True)
X_cen_test=X_test_matrix-means_test
S_test=(X_cen_test@X_cen_test.T)/999
eigen_val_test,eigen_vec_test=np.linalg.eigh(S_test)
sorted_indices_test = np.argsort(eigen_val_test)[::-1]
U_test = eigen_vec_test[:, sorted_indices_test]
Y_test=U_test.T@X_cen_test
UpX_test={}
for p in p_values:
  Up=U_test[:,:p]
  UpX_test[p]=Up.T@X_cen_test
x_test_pca_dfs={}
for p, UpX in UpX_test.items():
  x_test_pca_dfs[p]=pd.DataFrame(UpX.T)
y_pred_pca={}
for p in p_values:
  qda_=QDA_clf()
  qda_.train(x_train_pca_dfs[p], y_train_pca)
  y_pred_pca[p]=qda_.predict(x_test_pca_dfs[p])
for l,y in y_pred_pca.items():
  print(f'{l}: {y}')
total_test_samples=len(y_test_l)
for p in p_values:
  correct_pred=0
  for pred_l,true_l in zip(y_pred_pca[p],y_test_l):
    if pred_l==true_l: correct_pred+=1
  accuracy=correct_pred/total_test_samples
  print(f'Acuuracy Of QDA: {accuracy*10}')
print()




# %%
