# :mag_right: SCoPE: Sentence Content Paragraph Embeddings

**SCoPE**: **S**entence **Co**ntent **P**aragraph **E**mbeddings

Code and data for our ACL 2019 paper "[_Encouraging Paragraph Embeddings to Remember Sentence Identity Improves Classification_](https://www.aclweb.org/anthology/papers/P/P19/P19-1638/)".


## Installation prerequisites

This codebase is based on Yizhe Zhang's [implementation](https://github.com/dreasysnail/textCNN_public) of the CNN-R model (CNN-DCNN in the original [paper](https://arxiv.org/abs/1708.04729)) which requires Python version 2.7 and TensorFlow version 1.2.


## Probe task experiments


### 1. Hotel Reviews corpus

Paragraphs in our probe task experiments were extracted from the Hotel Reviews corpus ([Li et al., 2015](https://arxiv.org/abs/1506.01057)), which has previously been used for evaluating the quality of paragraph embeddings ([Li et al., 2015](https://arxiv.org/abs/1506.01057); [Zhang et al., 2017](https://arxiv.org/abs/1708.04729)). The original corpus can be found at 

https://github.com/jiweil/Hierarchical-Neural-Autoencoder or https://github.com/dreasysnail/textCNN_public.

The dataset used in our probe task is available at [here](https://drive.google.com/file/d/1ipjxeANqNRpE3zdrAm69Rjxe1suIOZF4/view?usp=sharing).

Script to read the dataset:

```python
import cPickle
data = cPickle.load(open('hotel_reviews_sentence_content.p', 'rb'))

unsup_train, unsup_val, unsup_test = data[0], data[1], data[2] 
x_train, x_val, x_test = data[3], data[4], data[5]
y_train, y_val, y_test = data[6], data[7], data[8]
token2index, index2token = data[9], data[10]
```


### 2. Probing paragraph embeddings for sentence content

We use Yizhe Zhang's [implementation](https://github.com/dreasysnail/textCNN_public) (`demo.py`) to train the CNN-R model with different numbers of dimensions for a maximum of 20 epochs with early stopping based on validation BLEU. After that, a classifier is trained on top of the frozen pre-trained CNN encoder for a maximum of 100 epochs with early stopping based on validation performance. The paragraph representation is computed either by extracting the bottleneck layer, i.e., **CNN-R** or performing average pooling over the learned word representations, i.e., **BoW(CNN-R)**.

Run `probe_CNN-R.py` and `probe_BoW-CNN-R.py` to reproduce our probe task results for the CNN-R and BoW(CNN-R) models, respectively.

Usage:

```
probe_CNN-R.py \
  --data-path DATA_PATH \
  --model-archive-path MODEL_ARCHIVE_PATH \
  --log-path LOG_PATH \
  --save-path SAVE_PATH \
  --from-scratch False \
  --unfrozen False \
  --embed-dim EMBED_DIM \
  --output-dim OUTPUT_DIM \
  --learning-rate LEARNING_RATE \
  --batch-size BATCH_SIZE \
  --num-epochs NUM_EPOCHS \
  --dropout-keep-prob DROPOUT_KEEP_PROB \
  [--print-freq PRINT_FREQ] \
  [--valid-freq VALID_FREQ]
```

```
probe_BoW-CNN-R.py \
  --data-path DATA_PATH \
  --model-archive-path MODEL_ARCHIVE_PATH \
  --log-path LOG_PATH \
  --save-path SAVE_PATH \
  --embed-dim EMBED_DIM \
  --learning-rate LEARNING_RATE \
  --batch-size BATCH_SIZE \
  --num-epochs NUM_EPOCHS \
  --dropout-keep-prob DROPOUT_KEEP_PROB \
  [--print-freq PRINT_FREQ] \
  [--valid-freq VALID_FREQ]
```


##  Paragraph classification experiments


### 3. Paragraph classification data

We experiment on three standard paragraph classification datasets: Yelp Review Polarity (Yelp), DBPedia, and Yahoo! Answers (Yahoo) ([Zhang et al., 2015](https://arxiv.org/abs/1509.01626)), which are instances of common classification tasks, including sentiment analysis and topic classification. These datasets are publicly available [here](http://goo.gl/JyCnZq). A preprocessed version of the Yelp dataset is available [here](https://drive.google.com/open?id=1qKos_wB45MzMu7Sn8RdvE6SRVAKCTC6e).

In an analysis experiment in the Appendix, we additionally experiment with the IMDB dataset ([Maas et al., 2011](https://aclweb.org/anthology/papers/P/P11/P11-1015/)) which can be downloaded from  http://ai.stanford.edu/~amaas/data/sentiment/ or this [direct link](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).

Use `generate_sentence_content_data.py` to generate sentence content data for pretraining.

Usage:

```
python generate_sentence_content_data.py \
  --dataname DATANAME \ # e.g., yelp, dbpedia, yahoo, imdb
  --data-path DATA_PATH \
  --use-all-sentences USE_ALL_SENTENCES \ # whether or not to create a pair of examples 
  # from every sentence in the paragraph to maximize the training data 
  # default: True
  --setting SETTING \ # how to sample negative sentence candidates: randomly ('rand'), 
  # or from paragraphs of the same class label as the probe paragraph ('in'),
  # or from paragraphs from a different class label ('out') 
  # default: 'rand'
```


### 4. Pretraining and finetuning CNN-SC

A visualization of our semi-supervised approach can be seen in the figure below. We first pretrain the CNN encoder (shown as two copies with shared parameters) on unlabeled data using our sentence content objective. The encoder is then used for downstream classification tasks.

<p align="center">
  <img src="https://github.com/tuvuumass/SCoPE-test/blob/master/figs/fig1.png" width="70%" alt="A visualization of our semi-supervised approach">
</p>

First, run `pretrain_CNN-SC.py` to pretrain the CNN encoder with our sentence content objective on the unlabeled data of the downstream classification task. 

Usage:

```
pretrain_CNN-SC.py \
  --data-path DATA_PATH \
  --log-path LOG_PATH \
  --save-path SAVE_PATH \
  --from-scratch True \
  --unfrozen True \
  --embed-dim EMBED_DIM \
  --output-dim OUTPUT_DIM \
  --learning-rate LEARNING_RATE \
  --batch-size BATCH_SIZE \
  --num-epochs NUM_EPOCHS \
  --dropout-keep-prob DROPOUT_KEEP_PROB \
  [--patience PATIENCE] \
  [--print-freq PRINT_FREQ] \
  [--valid-freq VALID_FREQ]
```

Then, run `finetune-CNN-SC.py` to fine-tune the CNN-SC model for the downstream classification task.

Usage:

```
finetune-CNN-SC.py \
  --data-path DATA_PATH \
  --train-portion TRAIN_PORTION \
  --model-archive-path MODEL_ARCHIVE_PATH \
  --log-path LOG_PATH \
  --save-path SAVE_PATH \
  --from-scratch False \
  --unfrozen True \
  --embed-dim EMBED_DIM \
  --output-dim OUTPUT_DIM \
  --learning-rate LEARNING_RATE \
  --batch-size BATCH_SIZE \
  --num-epochs NUM_EPOCHS \
  --dropout-keep-prob DROPOUT_KEEP_PROB \
  [--print-freq PRINT_FREQ] \
  [--valid-freq VALID_FREQ]
```


## References

If you use this code for your work, please cite us:

```
@inproceedings{vu-iyyer-2019-encouraging,
    title = "Encouraging Paragraph Embeddings to Remember Sentence Identity Improves Classification",
    author = "Vu, Tu  and
      Iyyer, Mohit",
    booktitle = "Proceedings of the 57th Conference of the Association for Computational Linguistics (ACL)",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1638",
    pages = "6331--6338"
}
```
