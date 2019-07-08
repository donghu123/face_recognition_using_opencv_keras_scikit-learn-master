#### PART OF THIS CODE IS USING CODE FROM VICTOR SY WANG: https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/utils.py ####

import numpy as np






def img_to_encoding(images, model):
    
    images = images[...,::-1] # Color image loaded by OpenCV is in BGR mode. But Matplotlib displays in RGB mode.
    images = np.around(images/255.0, decimals=12)
    # https://stackoverflow.com/questions/44972565/what-is-the-difference-between-the-predict-and-predict-on-batch-methods-of-a-ker
                                                                                                               
    if images.shape[0] > 1:
        embedding = model.predict(images, batch_size = 128)
    else:
        embedding = model.predict_on_batch(images)
    
    embedding = embedding / np.linalg.norm(embedding, axis = 1, keepdims = True)
    
    return embedding
