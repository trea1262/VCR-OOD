# Causal Debiasing for Visual Commonsense Reasoning (Accepted by ICASSP 2025)
The code is for Causal Debiasing for Visual Commonsense Reasoning.
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

