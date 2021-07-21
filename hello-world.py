# Import Scikit to use Tree
from sklearn import tree

# Specify the features and labels for our test. Apples are 0, Oranges are 1
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1] 

# We create a classifier with a decision tree to compare and let scikit infer what fruit may be
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# We print the prediction and it succeeds
print (clf.predict([[150, 0]]))