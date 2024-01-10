from sklearn import tree


#Because we understand the data, this will be a supervised learning model

# The labels are as follow:
# Apple = 0
# Orange = 1
# Lemon = 2

# The features are as follow:
# Weight
# Texture -> Smooth = 0, Bumpy = 1
# Color -> Red = 0, Orange = 1, Green = 2

# Apples are usually red, have a smooth texture and are somewhat light in weight
# Oranges are orange in color, have a bumpy texture and can be heavy in weight'
# Lemons are green, have a bumpy texture, and can either be light or heavy in weight

#Major Steps in ML
# 1) Define the problem: Tell the difference between an apple, orange and lemon

# 2) Build the Dataset:

features = [[100, 0, 0], [110, 0, 0], [103, 0, 0], [120, 1, 1], [150, 1, 1], [130, 1, 1], [120, 1, 2], [121, 1, 2], [125, 1, 2], [120, 0, 2]]
labels = ["Apple", "Apple", "Apple", "Orange", "Orange", "Orange", "Lemon", "Lemon", "Lemon", "Apple"]

# 3) Train the Model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# 4) Evaluate the model
#Create a function that takes weight, texture and color as parameters
#The prediction is then assigned to a variable to be returned

def guessFruit(weight, texture, color):
    result = clf.predict([[weight, texture, color]])
    return result
    
    
# 5) Use the model
print("I predict that the fruit will be a:" , guessFruit(120, 0, 2))

#by Kevin