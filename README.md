# Image-Captioning 
## Data 
Loading flickr8k datasets from https://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b  into folder: /content/drive/My Drive/datasets/ 

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
                    

## Preprocessing
I take a step to save image_feature of all images from pretrained Resnet and save embedding matrix for vocabulary

Run preprocess.py to prepare data for train. ( This may take a while )

## Training 
Run train.py to train your self

## Testing
Run data.py to see the demo



## Results
 

## References
This code inspired by : . https://www.tensorflow.org/tutorials/text/image_captioning
                        . https://github.com/krishnaik06/Deployment-Deep-Learning-Model
