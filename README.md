# DeepGOA
a deep neural networks for protein function prediction through combining sequence and PPI features.
# Requirements

 torch (0.40)
 
 pandas (0.23.1)
 
 numpy (1.15.0)
 
 fire (0.1.3)
 
 torchvision (0.2.1)
 
 pickle
 
 fire
 
# Usage
- download dataset and our trained model

  You first download [dataset](https://drive.google.com/file/d/1yO765opfaD_jFav5qshsvGpG0cFVTIkZ/view?usp=sharing) and save the dataset to the DeepGOA path. Before you run ouor saved mdoel, you should download [saved_model](https://drive.google.com/file/d/1bWilGYFPxO52aUKCUlutA6uJHr2W8bMJ/view?usp=sharing).
- train model

  You can run the DeepGOA.py file to train DeepGOA and the model will be saved in checkpoints path. Before you run the Predict_DeepGOA.py file, you should move the model to the saved_model path or change the load path. If you want to tune some hyper-parameters, you can change some values of hyper-parameters in config.py in utils folder.

  The other details can see the paper and the codes.
 
# Citation

# License
This project is licensed under the MIT License - see the LICENSE.txt file for details
