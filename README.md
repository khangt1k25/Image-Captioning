# Image-Captioning 
## Data 
Loading flickr8k datasets from https://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b  into folder : /content/drive/My Drive/datasets/ 
```
Structure:
├─ data.py                               % data_loader 
├─ models.py                             % all structure model in here 
├─ preprocessing.py                      % preprocessing for npy file
├─ train.py                              % train file
└─ utils.py                              % additional function
├─ captions.py                           % test file, design to app
├─ app.py                                % demo file, run by flask
├─ static                                %  
├─ templates                             % web front end
└─ content/drive/My Drive/datasets       % store data, result, checkpoint in here   
    └─Flickr8k
        └─ Flickr8k_Dataset
            └─ FLicker8k_Dataset
            └─ FLicker8k_nunpy
            └─ FLicker8k_stats
        ...
    └─ modelcheckpoint
        └─ train
                    
```

## Preprocessing
Run command to save image npy file
```
python preprocessing.py
```
## Training
Train by yourself
```
python train.py
```
## Testing
I am making demo by flask, so u can try to use
```
python app.py
```

## References
This code inspired by : . https://www.tensorflow.org/tutorials/text/image_captioning
                        . https://github.com/krishnaik06/Deployment-Deep-Learning-Model
