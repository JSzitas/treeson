import treeson

# Example usage
targets = [1]
max_depth = 10
min_nodesize = 2
seed = 42

# Create an instance of MultitargetRandomForest
model = treeson.MultitargetRandomForest(targets, max_depth, min_nodesize, seed)

# Fit the model
data = [[0.5, 0.4, 0.6, 1.5, 1.2, 1.4, 1.6, 2.5, 2.4, 2.6, 3.5, 3.4, 3.6, 4.5, 4.4, 4.6],
        [1.5, 1.4, 1.4, 1.6, 0.5, 0.4, 0.6, 3.5, 3.4, 3.6, 2.5, 2.4, 2.6, 4.5, 4.4, 4.6],
	[1, 2, 3, 1, 2, 3, 1, 1, 1, 3, 1, 2, 3, 1, 2, 3],
	[4, 5, 6, 4, 5, 6, 4, 5, 3, 6, 4, 5, 6, 4, 5, 6]]
#model.fit(data, n_tree=100)

# Predict using the model
predictions = model.memoryless_predict(data, data, 100, False, 0)

# Calculate feature importance
importance = model.feature_importance(data, 100, False, 0)

#print(predictions)
print(importance)
