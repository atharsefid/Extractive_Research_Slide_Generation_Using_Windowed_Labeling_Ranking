# This repository is for [Extractive Research Slide Generation Using Windowed Labeling Ranking](https://www.aclweb.org/anthology/2021.sdp-1.11/) article published at the
Scientific Scholarly Processing (SDP) workshop.
# Preprocess data
1. Dowload the paper and slides from [here](https://drive.google.com/file/d/1xYHXYoQBa7DJVrq0ePly58ioq2EmmVG8/view).
2. data dir structure:
```  
all_test.txt: list of test files
all_train.txt: list of train files
all_val.txt:  list of validation files
test  train  val: directories containing the data
```

# Train Model
To train the model run:

```$CUDA_VISIBLE_DEVICES=0 python3 batch_train.py ```

If the ```data_files/vocab.txt``` and ```lookup.pkl``` do not exist it will 
create the vocab file and the lookup table. 

Note: To change the training mode from summarunner to windowed_summarunner:
change the batch_data_utils.py -> get_article_labels

# Test model

1. Change the ```cpt``` (checkpoint) variable in batch_test.py.
2. Set the name of the the output
file ```e.g. predicted_windowed_summarunner```.
3. Run
```$CUDA_VISIBLE_DEVICES=2 python3 batch_test.py```

# Ecaluate the results
run ```python3 evaluateROUGE155.py```