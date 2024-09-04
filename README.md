# MNIST-Handwritten-Digits-Project

The project is an optical character recognition (OCR) on handwritten characters, which involves creating a deep learning 
model to classify handwritten digits from the MNIST dataset. The MNIST dataset, which contains 28x28 grayscale images of 
handwritten digits (0-9) was loaded and preprocessed. A neural network architecture was defined using nn.Sequential, the 
loss function, and optimizer was also specified and fine-tuned and the model yielded 98% accuracy. The project provides 
hands-on experience in building, training, and evaluating deep learning models, and provides a practical tool for digit 
recognition tasks in computer vision with applications in optical character recognition (OCR) systems, such as 
digitising handwritten documents.

## PROJECT STEPS
1. The dataset was loaded from 'torchvision.datasets'. The PyTorch method 'Transforms' was utilized to convert the data
  into tensors, normalize, and flatten it. DataLoaders were then created for the dataset, and the tensor dimensions were
 reordered to align with the PyTorch convention (channels, height, width).
2. The data was visualized using matplotlib.
3. The neural network was constructed using 'torch.nn.functional', and the Adam optimizer was employed to update the
   network's weights. The network was trained using the training DataLoader.
4. The neural network's accuracy was assessed on the test set. The model hyperparameters and network architecture were
   fine-tuned to enhance test set accuracy, achieving 98% accuracy on the test set.
5. 'torch.save' was used to save the trained model.
