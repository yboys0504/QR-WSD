# Quantum-inspired Representation Method for Long-tail Senses of Word Sense Disambiguation


## Work Description
Data imbalance, also known as the long-tailed distribution of data, is very common in the real world and is an important challenge for data-driven algorithms. Due to the long tail phenomenon of word sense distribution in linguistics, it is difficult to learn accurate representations for Low-Frequency Senses (LFSs) in Word Sense Disambiguation (WSD) tasks. Without data augmentation, exploring representation learning methods that do not rely on large sample sizes is an important means to combat the long tail. In this paper, inspired by the superposition state in quantum mechanics, a representation method in Hilbert space is proposed to reduce the dependence on a large sample size. We theoretically prove the effectiveness of this method; moreover, we verify the effect of the model based on the representation method under the WSD task, and it achieves state-of-the-art performance.


## Model Structure Diagram
<img src="https://github.com/yboys0504/QR-WSD/blob/main/chart.png">


## File And Folder Description
<b>data:</b> The data folder contains the training datasets. Due to github's restrictions on uploading files, here we give the link address of the datasets.

---<b>SemCor:</b> <a href="http://lcl.uniroma1.it/wsdeval/training-data">http://lcl.uniroma1.it/wsdeval/training-data</a>

---<b>OMSTI:</b> <a href="http://lcl.uniroma1.it/wsdeval/training-data">http://lcl.uniroma1.it/wsdeval/training-data</a>

---<b>Multilingual datasets:</b> <a href="https://github.com/SapienzaNLP/mwsd-datasets">https://github.com/SapienzaNLP/mwsd-datasets</a>


<b>ckpt:</b> The ckpt folder contains the pre-training code for the model.


<b>wsd_models:</b> The wsd_models folder contains two files, namely util.py and models.py.

---<b>util.py</b> contains the tool functions required by the main.py file; 

---<b>models.py</b> is the definition file of the model.


<b>main.py</b> is the entry file of the model, that is, the main class.


## Dependencies 
To run this code, you'll need the following libraries:
* [Python 3](https://www.python.org/)
* [Pytorch 1.2.0](https://pytorch.org/)
* [Transformers 1.1.0](https://github.com/huggingface/transformers)
* [Numpy 1.17.2](https://numpy.org/)
* [NLTK 3.4.5](https://www.nltk.org/)
* [tqdm](https://tqdm.github.io/)

We used the [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/) for training and evaluating our model.


## How to Run 
To train a biencoder model, run `python biencoder.py --data-path $path_to_wsd_data --ckpt $path_to_checkpoint`. The required arguments are: `--data-path`, which is the filepath to the top-level directory of the WSD Evaluation Framework; and `--ckpt`, which is the filepath of the directory to which to save the trained model checkpoints and prediction files. The `Scorer.java` in the WSD Framework data files needs to be compiled, with the `Scorer.class` file in the original directory of the Scorer file.

It is recommended you train this model using the `--multigpu` flag to enable model parallel (note that this requires two available GPUs). More hyperparameter options are available as arguments; run `python biencoder.py -h` for all possible arguments.

To evaluate an existing biencoder, run `python biencoder.py --data-path $path_to_wsd_data --ckpt $path_to_model_checkpoint --eval --split $wsd_eval_set`. Without `--split`, this defaults to evaluating on the development set, semeval2007. The model weights and predictions for the biencoder reported in the paper can be found [here](https://drive.google.com/file/d/1NZX_eMHQfRHhJnoJwEx2GnbnYIQepIQj).

Similar commands can be used to run the frozen probe for WSD (`frozen_pretrained_encoder.py`) and the finetuning a pretrained, single encoder classifier for WSD (`finetune_pretrained_encoder.py`).






