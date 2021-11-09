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



