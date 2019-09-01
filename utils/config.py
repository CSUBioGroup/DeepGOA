#-*- encoding:utf8 -*-

class DefaultConfig(object):
    
    #path
    saved_models_path = "./checkpoints"
    
    interPro_size = {"mf":[(1024,512)],"bp":[(1024,512)],"cc":[(1024,512)]}
    

    #train and test
    train_samples_num = {"cc":35546,"bp":36380,"mf":25224}
    cluster_samples_num = {"cc": 33557, "mf": 24766, "bp": 34839}

    
    #parameter setting
    batch_size = 128  #batch size
    use_gpu = True 
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4
    class_num = 2
    splite_rate = 0.8  #train
    THREADHOLD = 0.2
    
    
    #net setting
    lstm_hidden = 64
    kernels = [13,15,17]
    kernel_size = (8,16,32)
    hidden_channels = 64
    class_nums = {'mf':589,'bp':932,'cc':439}
    in_channel = 128
    features_W = 128
    features_L = 1000
    dropout = 0.2
    #net embedding parammeter
    embedding_num = 256


    
    
