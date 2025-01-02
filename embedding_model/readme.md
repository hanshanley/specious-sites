## MPNet Embedding Model
We provide the weights for the embedding model that we utilize in this work here. Specifically, we utilize a fine-tuned version of MPNet, optimized for semantic text similarity (STS) as described in Gao et al. This fine-tuning employs unsupervised contrastive learning for sentence embeddings and is conducted on a random sample of passages from our websites collected in January 2022. The process follows the default hyperparameters outlined by Gao et al., including a learning rate of 
3×10^-5, a batch size of 128, and 1M examples. Additionally, we freeze all but the last two layers of a publicly available MPNet model during fine-tuning.

The base `constrative_base.py` file contains code for loading our model and the `embeddings.py` file contains code for embedding texts. 

The weights for the model can be downloaded from [here](https://drive.google.com/file/d/1XAkI3sVlao2LJDGWGwTA0wnVdNff0Fli/view?usp=sharing).
