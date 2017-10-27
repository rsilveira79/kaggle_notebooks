## Models with facebook FastText

# Model 2
fasttext supervised -input spooky.train -output model_2_subword -epoch 30000 -lr 0.9 -wordNgrams 3 -loss ns -pretrainedVectors /home/ubuntu/repo/vectors/wiki-news-300d-1M-subword.vec -dim 300
Loss:

