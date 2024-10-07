import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
    # Fill missing values for numerical columns with mean
    for column in numeric_columns:
        if df[column].isnull().sum() != 0:
            df[column].fillna(df[column].mean(), inplace=True)

    # Fill missing values for categorical columns with the most frequent class
    for column in categorical_columns:
        if df[column].isnull().sum() != 0:
            df[column].fillna(df[column].mode(), inplace=True)
    
    return df

def convert_to_numeric(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    columns_to_encode = categorical_columns
    dfs_to_concat = []

    # Iterate through object columns
    for column in columns_to_encode:
        unique_values_count = df[column].nunique()

        if unique_values_count == 2:
            label_binarizer = LabelBinarizer()
            # Fit on the training set
            df[column] = label_binarizer.fit_transform(df[column])

        elif 2 < unique_values_count < df.shape[0]:
            # Perform one-hot encoding using pd.get_dummies
            one_hot_encoded = pd.get_dummies(df[column], prefix=column)
            one_hot_encoded = one_hot_encoded.astype('int')
            # Append the resulting DataFrame to the list
            dfs_to_concat.append(one_hot_encoded)
            df.drop(column, axis=1, inplace=True)

    # Concatenate all DataFrames in the list
    df = pd.concat([df] + dfs_to_concat, axis=1)
    return df


def NormalizeData(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    # for column in df.columns:
    #     if column in numeric_columns:
    #         df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    # Identify columns with a number datatype and more than 2 unique values
    # numerical_columns = df.select_dtypes(include=['number']).columns
    # numeric_columns = [col for col in numeric_columns if df[col].nunique() > 2]

    # Extract the selected columns and perform Z-score normalization
    # if the numeric colums has only 2 unique values and the values are 0 and 1 then exclude it from numeric colums
    for column in df.columns:
        if column in numeric_columns:
            if df[column].nunique() == 2 and df[column].min() == 0 and df[column].max() == 1:
                numeric_columns.remove(column)
            elif df[column].nunique ==1:
                numeric_columns.remove(column)
    
    if(len(numeric_columns) == 0):
        handle_missing_values(df)
        return df
    zscore_scaler = StandardScaler()
    # X_normalized = pd.DataFrame(zscore_scaler.fit_transform(X_selected), columns=X_selected.columns)
    
    df[numeric_columns] = zscore_scaler.fit_transform(df[numeric_columns])


    # Replace the original values in the DataFrame with the normalized values
    # df[numeric_columns] = X_normalized

    # Handle NaN after normalizing
    handle_missing_values(df)
    return df


def preprocessing(df):
    # fill null values. if data in that column is numeric, fill with mean. if data in that column is categorical, fill with mode
    for col in df.columns:
        # check if columns contains numeric data and convert it to numeric
        if df[col].dtype == 'object' and  df[col].str.isnumeric().sum() != 0:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = handle_missing_values(df)
    df = NormalizeData(df)
    df = convert_to_numeric(df)
    return df

# ==========================feature selection==========================

def entropy(series):
    length = series.shape[0]
    series = series.squeeze()
    values, counts = np.unique(series, return_counts=True)
    probabilities = counts / length
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Adding a small epsilon to avoid log(0)

def remainder(X_train, y_train, feature):

    total_entropy = 0

    for value in X_train[feature].unique():
        subset_indices = np.where(X_train[feature] == value)[0]
        subset = y_train.iloc[subset_indices,:]
        length = y_train.shape[0]
        total_entropy += len(subset) / length * entropy(subset)

    return total_entropy

def information_gain(X_train, y_train, feature):
    original_entropy = entropy(y_train)
    feature_remainder = remainder(X_train, y_train, feature)
    return original_entropy - feature_remainder

def select_k_best_features(X_train, y_train, k):
    features = X_train.columns
    gains = [(feature, information_gain(X_train, y_train, feature)) for feature in features]

    # Sort features based on information gain in descending order
    sorted_features = sorted(gains, key=lambda x: x[1], reverse=True)

    # Select the top k features
    selected_features = [feature for feature, _ in sorted_features[:k]]
    print(selected_features)
    selected_features_num = [X_train.columns.get_loc(feature) for feature in selected_features]

    return selected_features_num

# ==========================feature selection==========================

# write a logistic regression model for weak learner and to fit the data
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000, fit_intercept=False, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        epsilon = 1e-15  # Small value to prevent log(0)
        h = np.clip(h, epsilon, 1 - epsilon)  # Clip predicted values to prevent log(0) or log(1)
        loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        return loss
        # return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y, sample_weight,threshold=0):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        y = y.squeeze()

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        
        # gradient descent
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y) * sample_weight) / y.size
            # gradient = np.dot(X.T, (h - y) ) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
            if(loss < threshold):
                break
            if(self.verbose == True and i % 10000 == 0):
                print(f'loss: {loss} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X)
        return self.predict_prob(X) >= threshold


def choose_random_subset_with_weights(X, y,weights):

    # sample size is also random
    sample_size = np.random.randint(int(len(X)*0.3), len(X) + 1)
    # Get the indices for the random subset
    subset_indices = np.random.choice(len(X), size=sample_size, replace=False)
            
    subset_X = X.iloc[subset_indices, :]  # Subset of input features
    subset_y = y.iloc[subset_indices, :]  # Subset of target labels
    subset_weights = weights[subset_indices]

    return subset_X, subset_y, subset_weights


class AdaBoost:
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, _ = X.shape
        weights = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_estimators):
            X_subset, y_subset, weights_subset = choose_random_subset_with_weights(X, y, weights)
            # X_subset, y_subset, weights_subset = X, y, weights
            model = LogisticRegression()  # Replace DecisionStump with your weak learner (e.g., decision tree stump)
            model.fit(X_subset, y_subset, weights_subset,0)
            y_subset = y_subset.squeeze()
            predictions = model.predict(X,0.5)
            error = np.sum(weights * (np.sign(predictions) != y.squeeze()))

            if(error > 0.5):
                continue
            alpha = np.log((1.0 - error) / max(error, 1e-10))  # Avoid division by zero
            self.alphas.append(alpha)

            # Update weights where the predictions match with the actual labels , with the formula weights=wieghts*error/1-error
            for i in range(len(weights)):
                if predictions[i] == y[i]:
                    weights[i] *= (error / (1 - error))
            # Normalize weights
            weights /= np.sum(weights)

            self.models.append(model)
            print("done")

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            predictions += alpha * model.predict(X,0.5)

        return np.sign(predictions)



def testing_logistic(test_data,test_labels,logisticRegression):
    # Calculate the accuracy of the model on the test data
    # logisticRegression.fit(train_data, train_labels, None)
    prediction = logisticRegression.predict(test_data, 0.5)
    prediction = np.where(prediction > 0.5, 1, 0)
    accuracy = np.sum(prediction == test_labels.squeeze()) / len(test_labels)
    # print("np.sum ",np.sum(test_labels.squeeze() == 1))

    true_positive_rate = np.sum((prediction == test_labels.squeeze()) & (prediction == 1)) / np.sum(test_labels.squeeze() == 1)
    # true_positive_rate = np.sum(prediction == test_labels.squeeze() and prediction == 1) / np.sum(test_labels.squeeze() == 1)
    true_negative_rate = np.sum((prediction == test_labels.squeeze()) & (prediction == 0)) / np.sum(test_labels.squeeze() == 0)

    # print("np.sum ",np.sum(test_labels.squeeze() == 1))
    true_positive = np.sum((prediction == test_labels.squeeze()) & (prediction == 1))
    true_negative = np.sum((prediction == test_labels.squeeze()) & (prediction == 0))
    false_positive = np.sum((prediction != test_labels.squeeze()) & (prediction == 1))

    print("true_positive ",true_positive)
    print("false_positive ",false_positive) 
    precision = true_positive / (true_positive + false_positive)
    false_discorvery_rate = 1 - precision
    F1_score = 2 * (precision * true_positive_rate) / (precision + true_positive_rate) 
    print("Accuracy :" ,accuracy)
    print("True positive rate :" ,true_positive_rate)
    print("True negative rate :" ,true_negative_rate)
    print("Precision :" ,precision)
    print("False discorvery rate :" ,false_discorvery_rate)
    print("F1 score :" ,F1_score)

# import data from csv
# adult = fetch_ucirepo(id=2) 
  
# # data (as pandas dataframes) 
# X = adult.data.features 
# y = adult.data.targets 
# print(y.shape)
# train_data, test_data, train_labels,test_labels = train_test_split(X,y, test_size=0.2, random_state=29)
# print(type(train_data))
# print(type(train_labels))
# print(type(test_data))
# print(type(test_labels))
# print(train_data.shape)



# # ==========================churn dataset==========================
# data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
# data.drop(columns=['customerID'], inplace=True)

# train, test = train_test_split(data, test_size=0.2, random_state=29)

# train_data = train.iloc[:, :-1]
# train_labels = train.iloc[:, [-1]]
# test_data = test.iloc[:, :-1]
# test_labels = test.iloc[:, [-1]]

# train_data = preprocessing(train_data)
# train_labels = preprocessing(train_labels)
# test_data = preprocessing(test_data)
# test_labels = preprocessing(test_labels)

# # ==========================churn dataset==========================



# # ==========================adult dataset==========================

# column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship','race',
#                 'sex','capital-gain','capital-loss','hours-per-week','native-country','label']
# train = pd.read_csv('adult.data', header=None,sep=', ', engine='python',names=column_names)

# for i, j in train.iterrows():
#     for index,item in enumerate(j):
#         if str.strip(str(item)) == '?':
#             train.at[i,train.columns[index]] = train[train.columns[index]].mode()[0]    


# test = pd.read_csv('adult.test', header=None,sep=', ', engine='python',names=column_names)
# for i, j in test.iterrows():
#     for index,item in enumerate(j):
#         if str.strip(str(item)) == '?':
#             test.at[i,test.columns[index]] = test[test.columns[index]].mode()[0]


# train_data = train.iloc[:, :-1]
# train_labels = train.iloc[:, [-1]]
# test_data = test.iloc[:, :-1]
# test_labels = test.iloc[:, [-1]]

# train_data = preprocessing(train_data)
# train_labels = preprocessing(train_labels)
# test_data = preprocessing(test_data)
# test_labels = preprocessing(test_labels)
# # ==========================adult dataset==========================


# ==========================credit dataset==========================
data = pd.read_csv('creditcard.csv')
# randomly select 20000 samples where last column is the label and the lebel is 0 , and choose all the samples where the label is 1
random_samples_0 = data[data.iloc[:, -1] == 0].sample(n=20000, random_state=29)
samples_1 = data[data.iloc[:, -1] == 1]

data = pd.concat([random_samples_0, samples_1], ignore_index=True)

train, test = train_test_split(data, test_size=0.2, random_state=29)

train_data = train.iloc[:, :-1]
train_labels = train.iloc[:, [-1]]
test_data = test.iloc[:, :-1]
test_labels = test.iloc[:, [-1]]

train_data = preprocessing(train_data)
train_labels = preprocessing(train_labels)
test_data = preprocessing(test_data)
test_labels = preprocessing(test_labels)
# train_labels.to_csv('train_labels.csv', index=False)
# test_labels.to_csv('test_labels.csv', index=False)
# ==========================credit dataset==========================


k = train_data.shape[1]
if k > 15:
    k = int (k * 65 / 100)
selected_features = select_k_best_features(train_data, train_labels, k)
# exclude selected features if the numberof the index is greater than equal to test_data.shape[1]
selected_features = [feature for feature in selected_features if feature < test_data.shape[1]]

# Keep only k best features
train_data = train_data.iloc[: ,selected_features]
test_data = test_data.iloc[: ,selected_features]


logisticRegression = LogisticRegression()
logisticRegression.fit(train_data, train_labels, None,0) 
print("printing test data result")
testing_logistic(test_data,test_labels,logisticRegression)
print("printing train data result")
testing_logistic(train_data,train_labels,logisticRegression)


for i in range(5,21,5):
    adaBoost = AdaBoost()
    adaBoost.n_estimators = i
    adaBoost.fit(train_data, train_labels)

    prediction = adaBoost.predict(test_data)
    prediction = np.where(prediction > 0.5, 1, 0)
    accuracy = np.sum(prediction == test_labels.squeeze()) / len(test_labels)
    print("Printing accuracy of the AdaBoost model on test data for estimate:",i)
    print(accuracy)

    prediction = adaBoost.predict(train_data)
    prediction = np.where(prediction > 0.5, 1, 0)
    accuracy = np.sum(prediction == train_labels.squeeze()) / len(train_labels)
    print("Printing accuracy of the AdaBoost model on train data for estimate:",i)
    print(accuracy)





