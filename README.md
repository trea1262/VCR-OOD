# Causal Debiasing for Visual Commonsense Reasoning (Accepted by ICASSP 2025)
The code is for Causal Debiasing for Visual Commonsense Reasoning.
#### Paper
Visual Commonsense Reasoning (VCR) refers to answering questions and providing explanations based on images. While existing methods achieve high prediction accuracy, they often overlook bias in datasets and lack debiasing strategies. In this paper, our analysis reveals co-occurrence and statistical biases in both textual and visual data. We introduce the VCR-OOD datasets, comprising VCR-OOD-QA and VCR-OOD-VA subsets, which are designed to evaluate the generalization capabilities of models across two modalities. Furthermore, we analyze the causal graphs and prediction shortcuts in VCR and adopt a backdoor adjustment method to remove bias. Specifically, we create a dictionary based on the set of correct answers to eliminate prediction shortcuts. Experiments demonstrate the effectiveness of our debiasing method across different datasets.

#### Requirements
```
conda install numpy pyyaml setuptools cmake cffi tqdm pyyaml scipy ipython mkl mkl-include cython typing h5py pandas nltk spacy numpydoc scikit-learn jpeg
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
pip install -r allennlp-requirements.txt
pip install --no-deps allennlp==0.8.0
python -m spacy download en_core_web_sm
```

#### Data
Follow the steps in `data/README.md`. This includes the steps to get the pretrained BERT embeddings and the parsed results of sentences.

#### Train/Evaluate models

- For question answering, run:
```
python train_vcr.py -params models/multiatt/default_VCR.json -folder results/answer_save -train -test
```

- for Answer justification, run
```
python train_vcr.py -params models/multiatt/default_VCR.json -folder results/reason_save -train -test -rational
```

You can combine the validation predictions using
`python eval_q2ar.py`

