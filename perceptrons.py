############################################################
# CIS 521: Perceptrons Homework
############################################################

student_name = "Noah Eagle"

############################################################
# Imports
############################################################

import perceptrons_data as data

# Include your imports here, if any are used.

import math

############################################################
# Section 1: Perceptrons
############################################################

# A perceptron which performs binary classification (distinguishes between
# positive and negative instances)
class BinaryPerceptron(object):

    # Initialize the Binary Perceptron by training its weight vector on the
    # provided set of examples for a specified number of iterations
    def __init__(self, examples, iterations):

        # Initialize the weight vector (dictionary of feature to value mappings)
        weight_vector = {}

        # For each desired iteration of training...
        for i in range(iterations):

            # For each example data in our training set...
            for example in examples:

                # Get the data_vector and binary +/- label for this example
                data_vector, label = example

                # Initialize the dot product of weight and data vector to 0
                weight_data_dot_product = 0

                # For each feature in the data vector...
                for feature in data_vector:

                    # If the feature isn't in the weight vector...
                    if feature not in weight_vector:

                        # Add it to the weight vector and give it a value of 0
                        weight_vector[feature] = 0

                    # Compute the product of this feature's weighted value and 
                    # data value and add it to the running total for the
                    # weight and data vector dot product
                    weight_data_dot_product += weight_vector[feature] * data_vector[feature]

                # If the weight and data vector dot product is positive...
                if weight_data_dot_product > 0:

                    # We're predicting a True label
                    prediction = True

                # Otherwise (if it was 0 or negative)...
                else:

                    # We're predicting a False label
                    prediction = False

                # If our prediction was incorrect...
                if prediction != label:

                    # If the label was positive...
                    if label:

                        # Then update the weight vector by adding to it the
                        # data vector
                        for feature in data_vector:
                            weight_vector[feature] += data_vector[feature]

                    # Otherwise (if the label was negative)...
                    else:

                        # Then update the weight vector by subtracting from it
                        # the data vector
                        for feature in data_vector:
                            weight_vector[feature] -= data_vector[feature]

            # Save the trained weight vector into the w_vector field
            self.w_vector = weight_vector

    # Predicts the class of the provided example test (True or False) based on
    # the binary perceptron's current weight vector
    def predict(self, x):

        # Initialize the weight and data vector dot product to 0
        weight_data_dot_product = 0

        # For each feature in the test example
        for feature in x:

            # If the feature is in the weight vector (has a weight)
            if feature in self.w_vector:

                # Calculate the product of this feature's weight and its 
                # data value and add it to the running sum for the weight and
                # data vector dot product
                weight_data_dot_product += self.w_vector[feature] * x[feature]

        # If this weight and data vector dot product is positive, predict True
        if weight_data_dot_product > 0:
            return True

        # Otherwise (if it's 0 or negative), predict False
        else:
            return False

# A perceptron that performs multiclass classification (distinguishing
# between an arbitrary number of labeled groups)
class MulticlassPerceptron(object):

    # Initialize the Multiclass Perceptron by training its weight vectors on
    # the provided set of examples for a specified number of interations
    def __init__(self, examples, iterations):

        # Initialize a set to contain all the potential labels
        labels = set()

        # Initialize a dictionary (for mapping all features to 0)
        feature_dict = {}

        # For each example in the training set...
        for example in examples:

            # Get the data vector and label for this example
            data_vector, label = example

            # Add this label to the label set
            labels.add(label)

            # For each feature in the data vector...
            for feature in data_vector:

                # Make sure it's in the feature dictionary with a value of 0
                feature_dict[feature] = 0

        # Initialize a weights vector field (which will map labels to their
        # respective weight vectors)
        self.weight_vectors = {}

        # For each potential label...
        for label in labels:

            # Create a mapping from that label to a new dictionary in the
            # weights vector dictionary
            self.weight_vectors[label] = {}

            # For each possible feature...
            for feature in feature_dict:

                # Include a mapping from that feature to a value of 0 in this
                # label's weight vector
                self.weight_vectors[label][feature] = 0

        # For the desired number of iterations...
        for i in range(iterations):

            # For each example (data_vector, label) in the training set...
            for data_vector, label in examples:

                # Calculate which label the perceptron will predict for this
                # example data
                predicted_label = self.predict(data_vector)

                # If our predicted label doesn't match the actual label...
                if predicted_label != label:

                    # For every feature in the data vector...
                    for feature in data_vector:

                        # Take the correct label's weight vector feature value
                        # and add to it the data vector's feature value
                        self.weight_vectors[label][feature] += data_vector[feature]

                        # Take the incorrectly predicted label's weight vector
                        # feature value and subtract from it the data vector's
                        # feature value
                        self.weight_vectors[predicted_label][feature] -= data_vector[feature]
    
    # Predicts the class of the provided example test (could be any of our
    # known labels) based on the multiclass perceptron's current weight vectors
    def predict(self, x):

        # Initialize our current best label (predicted label) to None
        current_best_label = None

        # Initialize our current best label score (weight vector and data vector
        # dot product) to negative infinity
        current_best_score = -math.inf

        # For each label we have weight vectors for...
        for label in self.weight_vectors:

            # Initialize its weight vector and data vector dot product to 0
            weight_data_dot_product = 0

            # For every feature in the data example...
            for feature in x:

                # Increment the running weight and data vector dot product by
                # the product of the feature value from this label's weight
                # vector and the data value for this feature
                weight_data_dot_product += self.weight_vectors[label][feature] * x[feature]

            # If the weight and data vector dot product is greater than our
            # current best score...
            if weight_data_dot_product > current_best_score:

                # Update it to be our new current best score
                current_best_score = weight_data_dot_product

                # Update its corresponding label to be our new current best
                # label
                current_best_label = label

        # After going through all label weight vectors, return our best label
        return current_best_label

############################################################
# Section 2: Applications
############################################################

# A (Multiclass) Perceptron for distinguishing between three types of Irises
# (setosa, versicolor, and virginica)
class IrisClassifier(object):

    # Initializes a Multiclass Perceptron trained on Iris classification data.
    def __init__(self, data):

        # Initialize the training data list
        training_list = []

        # For each data entry...
        for entry in data:

            # Get the values and its label
            values, label = entry

            # Initialize a dictionary (to map features to values)
            values_dict = {}

            # Map each of the four Iris features to their corresponding
            # data values
            values_dict["x1"] = values[0]
            values_dict["x2"] = values[1]
            values_dict["x3"] = values[2]
            values_dict["x4"] = values[3]

            # Create a tuple of the (feature: value) dictionary and the label
            training_tuple_entry = (values_dict, label)

            # Append said tuple to the training data list
            training_list.append(training_tuple_entry)

        # Create a Multiclass Perceptron that trains on the training list
        # for 1000 iterations
        self.perceptron = MulticlassPerceptron(training_list, 1000)

    # Takes in a test data example for an Iris and predicts which type of Iris
    # it is
    def classify(self, instance):

        # Initialize the test (feature: value) dictionary
        test = {}

        # Map each of the four Iris features to their corresponding data values
        test["x1"] = instance[0]
        test["x2"] = instance[1]
        test["x3"] = instance[2]
        test["x4"] = instance[3]

        # Predict this test example's Iris type
        return self.perceptron.predict(test)

# A (Multiclass) Perceptron for distinguishing between digits 0-9 
class DigitClassifier(object):

    # Initializes a Multiclass Perceptron trained on digit classification data
    def __init__(self, data):

        # Initialize the training data list
        training_list = []

        # For each data entry...
        for entry in data:

            # Get the values and its label
            values, label = entry

            # Initialize a dictionary (to map features to values)
            values_dict = {}

            # For each of the 64 features...
            for i in range(64):

                # Map feature xi to its corresponding value from the data entry
                values_dict["x" + str(i)] = values[i]

            # Create a tuple of the (feature: value) dictionary and the label
            training_tuple_entry = (values_dict, label)

            # Append said tuple to the training data list
            training_list.append(training_tuple_entry)

        # Create a Multiclass Perceptron that trains on the training list for
        # 10 iterations
        self.perceptron = MulticlassPerceptron(training_list, 10)

    # Takes in a test data example for a digit and predicts which digit it is
    def classify(self, instance):

        # Initialize the test (feature: value) dictionary
        test = {}

        # For each of the 64 features...
        for i in range(64):

            # Map feature xi to its corresponding data value
            test["x" + str(i)] = instance[i]

        # Predict which digit this test example is
        return self.perceptron.predict(test)

# A (Binary) Perceptron for distinguishing between numbers > 1 and <= 1
class BiasClassifier(object):

    # Initializes a Binary Perceptron trained on single-feature numeric data
    def __init__(self, data):

        # Initialize the training data list
        training_list = []

        # For each data entry...
        for entry in data:

            # Get the value and its label
            value, label = entry

            # Initialize a dictionary (to map features to values)
            values_dict = {}

            # Map the single feature to the data value - 1 (so that things above
            # 1 remain positive (and thus True by our 0 threshold) while
            # things below 1 become negative (and thus False by our 0 
            # threshold))
            values_dict["x1"] = value - 1

            # Create a tuple of the (feature: value) dictionary and the label
            training_tuple_entry = (values_dict, label)

            # Append said tuple to the training data list
            training_list.append(training_tuple_entry)

        # Create a Binary Perceptron that trains on the training list for
        # 10 iterations
        self.perceptron = BinaryPerceptron(training_list, 10)

    # Takes in a test data example and predicts whether it should be labeled
    # True or False (based on whether it's above 1 or not)
    def classify(self, instance):

        # Initialize the test (feature: value) dictionary
        test = {}

        # Map the singular feature to the data value - 1 (to account for
        # threshold shift from 0 to 1)
        test["x1"] = instance - 1

        # Predict whether this number should be labeled True or False
        return self.perceptron.predict(test)

# A (Binary) Perceptron for distinguishing between a set of mystery data
# (which appears to be separable into two clases based on the two inputs'
# distance from the origin if you consider the inputs to be (x, y) coordinates)
class MysteryClassifier1(object):

    # Initializes a (Binary) Perceptron trained on the mystery1 data
    def __init__(self, data):

        # Initialize a training data set
        training_list = []

        # For each data entry...
        for entry in data:

            # Get the values and its label
            values, label = entry

            # Initialize a dictionary (to map features to values)
            values_dict = {}

            # For feature 1, give it a value of the distance of the coordinate
            # (input1, input2) from the origin
            values_dict["x1"] = math.sqrt((values[0] ** 2) + (values[1] ** 2))

            # Give feature 2 a constant value of 1 (an extra feature to allow
            # for some bias threshold shifting and so that the classification
            # doesn't depend on the distance being <= 0 or > 0)
            values_dict["x4"] = 1

            # Create a tuple of the (feature: value) dictionary and the label
            training_tuple_entry = (values_dict, label)

            # Append said tuple to the training data list
            training_list.append(training_tuple_entry)

        # Create a Binary Perceptron that trains on the training list for 
        # 10 iterations
        self.perceptron = BinaryPerceptron(training_list, 10)

    # Takes in a mystery test data example and predicts which of the two
    # classes it belongs to
    def classify(self, instance):

        # Initialize the test (feature: value) dictionary
        test = {}

        # Map the first feature to the distance of the coordinate 
        # (input1, input2) to the origin
        test["x1"] = math.sqrt((instance[0] ** 2) + (instance[1] ** 2))

        # Map the second feature to a constant value of 1 (for threshold bias
        # shifting)
        test["x4"] = 1

        # Predict which class this mystery data example belongs to
        return self.perceptron.predict(test)

# A (Binary) Perceptron for distinguishing between a set of mystery data
# (which appears to be divided into two classes based on whether the product
# of all inputs is positive or negative (so whether there are an odd or even
# number of negative inputs))
class MysteryClassifier2(object):

    # Initializes a (Binary) Perceptron trained on the mystery2 data
    def __init__(self, data):

        # Initialize a training data set
        training_list = []

        # For each data entry...
        for entry in data:

            # Get the values and its label
            values, label = entry

            # Initialize a dictionary (to map features to values)
            values_dict = {}

            # For feature 1, give it a value of the product of the three inputs
            values_dict["x1"] = values[0] * values[1] * values[2]

            # Give feature 2 a value of the first input
            values_dict["x2"] = values[0]

            # Give feature 3 a value of the second input
            values_dict["x3"] = values[1]

            # Give feature 4 a value of the third input
            values_dict["x4"] = values[2]

            # Create a tuple of the (feature: value) dictionary and the label
            training_tuple_entry = (values_dict, label)

            # Append said tuple to the training data list
            training_list.append(training_tuple_entry)

        # Create a Binary Perceptron that trains on the training list for
        # 100 iterations
        self.perceptron = BinaryPerceptron(training_list, 100)

    # Takes in a mystery test data example and predicts which of the two
    # classes it belongs to
    def classify(self, instance):

        # Initialize the test (feature: value) dictionary
        test = {}

        # Map the first feature to the product of the inputs
        test["x1"] = instance[0] * instance[1] * instance[2]

        # Map the second feature to the first input
        test["x2"] = instance[0]

        # Map the third feature to the second input
        test["x3"] = instance[1]

        # Map the fourth feature to the third input
        test["x4"] = instance[2]

        # Predict which class this mystery data example belongs to
        return self.perceptron.predict(test)