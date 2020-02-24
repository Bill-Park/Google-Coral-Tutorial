import numpy as np

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter, test_images, test_labels) :
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for test_image in test_images:
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_digits)):
    if prediction_digits[index] == test_labels[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_digits)

  return accuracy
  
 
def evaluate_single(interpreter, image) :
	input_index = interpreter.get_input_details()[0]["index"]
	output_index = interpreter.get_output_details()[0]["index"]
	
	image_32 = np.expand_dims(image, axis=0).astype(np.float32)
	interpreter.set_tensor(input_index, image_32)
	interpreter.invoke()
	
	predictions = interpreter.get_tensor(output_index)
	index = np.argmax(predictions[0])
	
	label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	return label[index]
	
	
	 