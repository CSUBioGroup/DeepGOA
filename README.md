# DeepGOA
A deep learning framework for gene ontology annotation with sequence- and network-based information.
# Requirements

 torch (0.4.0)
 
 pandas (0.23.1)
 
 numpy (1.15.0)
 
 fire (0.1.3)
 
 torchvision (0.2.1)
 
 pickle
 
# Usage
- download dataset and our trained model
You first need to download the source code and [dataset](http://bioinformatics.csu.edu.cn/resources/softs/zhangfuhao/?tdsourcetag=s_pcqq_aiomsg). Then you should extract the dataset to the DeepGOA directory. If you want to run our saved model, you should download [saved_model](http://bioinformatics.csu.edu.cn/resources/softs/zhangfuhao/?tdsourcetag=s_pcqq_aiomsg) and save to the DeepGOA directory.
  
- train model

  You can run the DeepGOA.py file to train DeepGOA and the model will be saved in checkpoints path. Before you run the Predict_DeepGOA.py file, you should move the model to the saved_model path or change the load path. If you want to tune some hyper-parameters, you can change some values of hyper-parameters in config.py in utils folder.

  The other details can see the paper and the codes.
 
# Citation

# License
This project is licensed under the MIT License - see the LICENSE.txt file for details
