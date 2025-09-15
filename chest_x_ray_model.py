
# coding: utf-8

# # Chest X-Ray Medical Diagnosis with Deep Learning


# Run the next cell to import all the necessary packages.

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from keras.models import load_model

import util
from public_tests import *
from test_utils import *

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# <a name='2'></a>
# ## 2. Load the Datasets


# To make your job a bit easier, we have processed the labels for our small sample and generated three new files to get you started. These three files are:
# 
# 1. `nih/train-small.csv`: 875 images from our dataset to be used for training.
# 1. `nih/valid-small.csv`: 109 images from our dataset to be used for validation.
# 1. `nih/test.csv`: 420 images from our dataset to be used for testing. 
# 
# This dataset has been annotated by consensus among four different radiologists for 5 of our 14 pathologies:
# - `Consolidation`
# - `Edema`
# - `Effusion`
# - `Cardiomegaly`
# - `Atelectasis`


# As long as you are aware of all this though, it should not cause you any confusion as the term 'class' is usually clear from the context in which it is used.

# <a name='2-1'></a>
# ### 2.1 Loading the Data
# Let's open these files using the [pandas](https://pandas.pydata.org/) library

# In[8]:


train_df = pd.read_csv("data/nih/train-small.csv")
valid_df = pd.read_csv("data/nih/valid-small.csv")

test_df = pd.read_csv("data/nih/test.csv")

train_df.head()


# In[5]:


labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']


# <a name='2-2'></a>
# ### 2.2 Preventing Data Leakage
# It is worth noting that our dataset contains multiple images for each patient. This could be the case, for example, when a patient has taken multiple X-ray images at different times during their hospital visits. In our data splitting, we have ensured that the split is done on the patient level so that there is no data "leakage" between the train, validation, and test datasets.



# In[6]:


# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """

    
    # Get unique patient IDs from both datasets
    
    df1_patients_unique = set(df1[patient_col].unique())
    df2_patients_unique = set(df2[patient_col].unique())
    
    # Find intersection
    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)

    # leakage contains true if there is patient overlap, otherwise false.
    leakage = len(patients_in_both_groups) > 0 # boolean (true if there is at least 1 patient in both groups)
    
    
    return leakage


# In[7]:


### do not edit this code cell    
check_for_leakage_test(check_for_leakage)


# ##### Expected output
# 
# ```Python
# Test Case 1
# 
# df1
#    patient_id
# 0           0
# 1           1
# 2           2
# df2
#    patient_id
# 0           2
# 1           3
# 2           4
# leakage output: True 
# -------------------------------------
# Test Case 2
# 
# df1
#    patient_id
# 0           0
# 1           1
# 2           2
# df2
#    patient_id
# 0           3
# 1           4
# 2           5
# leakage output: False
# ```
# ```
#  All tests passed.
# ```

# Run the next cell to check if there are patients in both train and test or in both valid and test.

# In[9]:


print("leakage between train and valid: {}".format(check_for_leakage(train_df, valid_df, 'PatientId')))
print("leakage between train and test: {}".format(check_for_leakage(train_df, test_df, 'PatientId')))
print("leakage between valid and test: {}".format(check_for_leakage(valid_df, test_df, 'PatientId')))


# ##### Expected output
# 
# ```Python
# leakage between train and valid: True
# leakage between train and test: False
# leakage between valid and test: False
# ```

# <a name='2-3'></a>
# ### 2.3 Preparing Images

# With our dataset splits ready, we can now proceed with setting up our model to consume them. 
# - For this we will use the off-the-shelf [ImageDataGenerator](https://keras.io/preprocessing/image/) class from the Keras framework, which allows us to build a "generator" for images specified in a dataframe. 


# In[10]:


def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator


# #### Build a separate generator for valid and test sets
# 
# Now we need to build a new generator for validation and testing data. 


# In[11]:


def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for validation set and test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="Image", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator


# With our generator function ready, let's make one generator for our training data and one each of our test and  validation datasets.

# In[13]:


IMAGE_DIR = "data/nih/images-small/"
train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)


# Let's peek into what the generator gives our model during training and validation. We can do this by calling the `__get_item__(index)` function:

# In[15]:


x, y = train_generator.__getitem__(4)
plt.imshow(x[0]);


# <a name='3'></a>
# ## 3. Model Development

# <a name='3-1'></a>
# ### 3.1 Addressing Class Imbalance

# In[16]:


plt.xticks(rotation=90)
plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
plt.title("Frequency of Each Class")
plt.show()


# We can see from this plot that the prevalance of positive cases varies significantly across the different pathologies. (These trends mirror the ones in the full dataset as well.) 



# In[17]:


# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # total number of patients (rows)
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    ### END CODE HERE ###
    return positive_frequencies, negative_frequencies


# In[18]:


### do not edit this code cell       
compute_class_freqs_test(compute_class_freqs)


# ##### Expected output
# 
# ```Python
# Labels:
# [[1 0 0]
#  [0 1 1]
#  [1 0 1]
#  [1 1 1]
#  [1 0 1]]
# 
# Pos Freqs:  [0.8 0.4 0.8]
# Neg Freqs:  [0.2 0.6 0.2] 
# ```
# ```
#  All tests passed. 
# ```

# Now we'll compute frequencies for our training data.

# In[19]:


freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
freq_pos


# ##### Expected output
# 
# ```Python
# array([0.02 , 0.013, 0.128, 0.002, 0.175, 0.045, 0.054, 0.106, 0.038,
#        0.021, 0.01 , 0.014, 0.016, 0.033])
# ```
# 
# 
# Let's visualize these two contribution ratios next to each other for each of the pathologies:

# In[20]:


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)


# As we see in the above plot, the contributions of positive cases is significantly lower than that of the negative ones. However, we want the contributions to be equal. One way of doing this is by multiplying each example from each class by a class-specific weight factor, $w_{pos}$ and $w_{neg}$, so that the overall contribution of each class is the same. 
# 
# To have this, we want 
# 
# $$w_{pos} \times freq_{p} = w_{neg} \times freq_{n},$$
# 
# which we can do simply by taking 
# 
# $$w_{pos} = freq_{neg}$$
# $$w_{neg} = freq_{pos}$$
# 
# This way, we will be balancing the contribution of positive and negative labels.

# In[21]:


pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights


# Let's verify this by graphing the two contributions next to each other again:

# In[22]:


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data);



# After computing the weights, our final weighted loss for each training case will be 

# ##### Note
# Please use Keras functions to calculate the mean and the log.
# 
# - [Keras.mean](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/backend/mean)
# - [Keras.log](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/backend/log)

# In[23]:


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
             loss += - K.mean(
                pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon) +
                neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon)
            )
        return loss
    
        ### END CODE HERE ###
    return weighted_loss


# In[24]:


# test with a large epsilon in order to catch errors. 
# In order to pass the tests, set epsilon = 1
epsilon = 1

### do not edit anything below
sess = K.get_session()
get_weighted_loss_test(get_weighted_loss, epsilon, sess)


# ##### Expected output
# 
# with epsilon = 1
# ```Python
# Your outputs:
# 
# L(y_pred_1) =  -0.4956203
# L(y_pred_2) =  -0.4956203
# Difference: L(y_pred_1) - L(y_pred_2) =  0.0 
# ```
# ```
#  All tests passed.   
# ```
# 
# If you are missing something in your implementation, you will see a different set of losses for `L(y_pred_1)` and `L(y_pred_2)` (even though `L(y_pred_1)` and `L(y_pred_2)` will be the same).

# <a name='3-2'></a>
# ### 3.2 DenseNet121
# 
# Next, we will use a pre-trained [DenseNet121](https://www.kaggle.com/pytorch/densenet121) model which we can load directly from Keras and then add two layers on top of it:
# 1. A `GlobalAveragePooling2D` layer to get the average of the last convolution layers from DenseNet121.
# 2. A `Dense` layer with `sigmoid` activation to get the prediction logits for each of our classes.
# 
# We can set our custom loss function for the model by specifying the `loss` parameter in the `compile()` function.

# In[25]:


# create the base pre-trained model
base_model = DenseNet121(weights='models/nih/densenet.hdf5', include_top=False)

x = base_model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))



# ## 4. Training (Optional)
# Since training can take a considerable time, for pedagogical purposes we have chosen not to train the model here but rather to load a set of pre-trained weights in the next section. However, you can use the code shown below to practice training the model locally on your machine or in Colab.
# 
# **NOTE:** Do not run the code below on the Coursera platform as it will exceed the platform's memory limitations.
# 
# Python Code for training the model:
# 
# ```python
# history = model.fit_generator(train_generator, 
#                               validation_data=valid_generator,
#                               steps_per_epoch=100, 
#                               validation_steps=25, 
#                               epochs = 3)
# 
# plt.plot(history.history['loss'])
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.title("Training Loss Curve")
# plt.show()
# ```

# <a name='4-1'></a>
# ### 4.1 Training on the Larger Dataset

# Given that the original dataset is 40GB+ in size and the training process on the full dataset takes a few hours, we have trained the model on a GPU-equipped machine for you and provided the weights file from our model (with a batch size of 32 instead) to be used for the rest of this assignment. 
# 
# The model architecture for our pre-trained model is exactly the same, but we used a few useful Keras "callbacks" for this training. Do spend time to read about these callbacks at your leisure as they will be very useful for managing long-running training sessions:


# You can read about these callbacks and other useful Keras callbacks [here](https://keras.io/callbacks/).
# 
# Let's load our pre-trained weights into the model now:

# In[28]:


model.load_weights("models/nih/pretrained_model.h5")


# <a name='5'></a>
# ## 5. Prediction and Evaluation

# Now that we have a model, let's evaluate it using our test set. We can conveniently use the `predict_generator` function to generate the predictions for the images in our test set.

# In[29]:


predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))


# <a name='5-1'></a>
# ### 5.1 ROC Curve and AUROC


# In[30]:


auc_rocs = util.get_roc_curve(labels, predicted_vals, test_generator)


# You can compare the performance to the AUCs reported in the original ChexNeXt paper in the table below: 

# First we will load the small training set and setup to look at the 4 classes with the highest performing AUC measures.

# In[31]:


df = pd.read_csv("data/nih/train-small.csv")
IMAGE_DIR = "data/nih/images-small/"

# only show the labels with top 4 AUC
labels_to_show = np.take(labels, np.argsort(auc_rocs)[::-1])[:4]


# Now let's look at a few specific images.

# In[32]:


util.compute_gradcam(model, '00008270_015.png', IMAGE_DIR, df, labels, labels_to_show)


# In[ ]:


util.compute_gradcam(model, '00011355_002.png', IMAGE_DIR, df, labels, labels_to_show)


# In[ ]:


util.compute_gradcam(model, '00029855_001.png', IMAGE_DIR, df, labels, labels_to_show)


# In[ ]:


util.compute_gradcam(model, '00005410_000.png', IMAGE_DIR, df, labels, labels_to_show)

