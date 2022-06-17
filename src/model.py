from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

def get_model(model_type):
	# TODO: More model types
	if model_type == "Logistic":
		return LogisticRegression(solver = "liblinear")

	elif model_type == "MLP":
		return MLPClassifier(
			solver = "lbfgs",
			activation = "relu",
			alpha = 0.1,
			random_state = 0,
			hidden_layer_sizes = [10, 10]
		)

	elif model_type == "Decision-tree":
		return DecisionTreeClassifier(max_depth = 4, random_state = 0)

	elif model_type == "Random-forest":
		return RandomForestClassifier(n_estimators = 10, random_state = 0)

	elif model_type == "Grad-tree":
		return GradientBoostingClassifier(random_state = 0)

	else:
		print("Unknown model type. Aborted.")
		raise Exception()

if __name__ == "__main__":
    print("Use this module by import-ing it.")