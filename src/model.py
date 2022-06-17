from sklearn.linear_model import LogisticRegression

def get_model(input_, label, model_type):
	# TODO: More model types
	if model_type == "Logistic":
		model = LogisticRegression(solver = "liblinear")
		model.fit(input_, label)
		return model

	else:
		print("Unknown model type. Aborted.")
		raise Exception()

if __name__ == "__main__":
    print("Use this module by import-ing it.")