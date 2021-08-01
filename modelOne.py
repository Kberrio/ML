from sklearn import tree
#sklearn is the module, and tree is a modile for clasification and regression

#Because we understand the data, this will be a supervised learning model

# Features are: smooth, light, bumpy and heavy
# smooth = 0, bumpy = 1


#Apple is smooth and light = 0
#Orange is bumpy and heavier = 1
#Limon is heavier than orange and has similar texture


#Color feature:
#Red will be 0
#Orange will be 1
#Green will be 2

# 1) Define the problem: Diferenciar manzanas a naranjas
# 2) Build the dataset


features = [[100, 0, 0], [110, 0, 0], [103, 0, 0], [120, 1, 1], [150, 1, 1], [130, 1, 1], [120, 1, 2], [121, 1, 2], [125, 1, 2], [120, 0, 2]]
labels = ["Manzana", "Manzana", "Manzana", "Naranja", "Naranja", "Naranja", "Limon", "Limon", "Limon", "Manzana"]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print ("Corriendo satisfactoriamente!")

print(clf.predict([[115, 0, 2]]))

