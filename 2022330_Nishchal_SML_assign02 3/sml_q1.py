
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from collections import defaultdict

class QDA:
  def __init__(self):
    self.estimates = None
  def train(self, X_train, y_train):
        self.estimates = self.compute_estimates(X_train, y_train)
  def compute_estimates(self, X_train, y_train):
    df = pd.concat([x_train, y_train], axis=1)
    df.set_index(y_train.columns[0], inplace=True)
    unique_labels=df.index.unique()
    estimates = []
    mean_vectors = {}
    ep=1e-6


    for label in unique_labels:
        label_rows = df.loc[df.index == label]  # Extract rows for the current label
        mean_vector = label_rows.mean(axis=0)   # Compute the mean vector along the columns
        mean_vectors[label] = mean_vector       # Store the mean vector for the current label
    covariance_matrices={}
    for label in unique_labels:
        label_rows = df.loc[df.index == label]  # Extract rows for the current label
        cov_matrix = label_rows.cov()           # Compute the covariance matrix
        covariance_matrices[label] = cov_matrix # Store the covariance matrix for the current label
    det_c = {}
    inv_cov_mat = {}

    # Iterate through each label and its corresponding covariance matrix
    for label, cov_matrix in covariance_matrices.items():
        # Add a small positive constant times the identity matrix to make the covariance matrix invertible
        cov_mat_reg = cov_matrix + ep * np.eye(cov_matrix.shape[0])

        # Compute the determinant of the regularized covariance matrix
        det = np.linalg.det(cov_mat_reg)

        # If the determinant is zero or infinity, use pseudoinverse to compute the inverse covariance matrix
        if det == 0 or np.isinf(det):
            det = np.finfo(float).eps
            inv_cov = np.linalg.pinv(cov_mat_reg)
        else:
            # Otherwise, compute the inverse covariance matrix
            inv_cov = np.linalg.inv(cov_mat_reg)

        # Store the determinant and inverse covariance matrix for the current label
        det_c[label] = det
        inv_cov_mat[label] = inv_cov

    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    class_priors = {}
    for c, count in zip(unique_classes, class_counts):
        prior = count / total_samples
        class_priors[c] = prior
    estimates.append((unique_labels,mean_vectors,covariance_matrices,inv_cov_mat,det_c,class_priors))
    return estimates

  def predict(self,x_test):
      predicted_labels = []
      unique_labels=self.estimates[0][0]
      mean_vectors=self.estimates[0][1]
      covariance_matrices=self.estimates[0][2]
      inv_cov_mat=self.estimates[0][3]
      det_c=self.estimates[0][4]
      class_priors=self.estimates[0][5]

      for index, row in x_test.iterrows():
        max_disc = float('-inf')  # Initialize maximum discriminant value
        predicted_label = None  # Initialize predicted label

        # Compute discriminant for each label
        for label, cov_matrix in covariance_matrices.items():
            mean_vector = mean_vectors[label]  # Get the mean vector for the current label
            inv_cov = inv_cov_mat[label]       # Get the inverse covariance matrix for the current label
            prior_prob = class_priors[label]    # Get the prior probability for the current label

            # Compute the quadratic term (X - mean_vector)^T * inv_cov * (X - mean_vector)
            quadratic_term = np.dot(np.dot((row - mean_vector), inv_cov), (row - mean_vector))

            # Compute the discriminant function
            discriminant = -0.5 * np.log(det_c[label]) - 0.5 * quadratic_term + np.log(prior_prob)

            # Update predicted label if the discriminant is higher
            if discriminant > max_disc:
                max_disc = discriminant
                predicted_label = label

        # Append the predicted label to the list
        predicted_labels.append(predicted_label)
      return predicted_labels

def class_idx(x):
  class_indices = {i: [] for i in range(10)}
  for i, label in enumerate(x):
    class_indices[label].append(i)
  return class_indices

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
mnist_data = mnist.load_data(path="mnist.npz")
(train_images, train_labels), (test_images,test_labels) = mnist_data
train_images=train_images.astype(np.float64)/255
test_images=test_images.astype(np.float64)/255
# print((train_images.shape,test_images.shape))

image_plot(x_img=train_images,x=train_labels)
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

qda_clf=QDA()
qda_clf.train(x_train,y_train)
y_pred=qda_clf.predict(x_test)
y_test_l=list(y_test)
total_test_samples=len(y_test_l)
correct_pred=0
for pred_l,true_l in zip(y_pred,y_test_l):
  if pred_l==true_l: correct_pred+=1
accuracy=correct_pred/total_test_samples
print(f'Acuuracy Of QDA: {accuracy}')

class_correct = defaultdict(int)
class_total = defaultdict(int)

# Iterate through each predicted and true label pair
for pred_label, true_label in zip(y_pred, y_test):
    # Update the total count of samples for the true label
    class_total[true_label] += 1
    # Check if the predicted label matches the true label
    if pred_label == true_label:
        # Increment the count of correct predictions for the true label
        class_correct[true_label] += 1

# Initialize a dictionary to store the accuracy of each class
class_accuracy = {}

# Calculate the accuracy for each class
for label in class_total.keys():
    # Calculate the accuracy for the current class
    if class_total[label] > 0:
        class_accuracy[label] = class_correct[label] / class_total[label]
    else:
        class_accuracy[label] = 0.0

# Print the accuracy of each class
for label, accuracy in class_accuracy.items():
    print(f"Accuracy for class {label}: {accuracy}")



# %%
