# Data
###################################################VCR-OOD#################################################################################################################
In the VCR-OOD dataset, the QA and VA subsets share a train set, and their test set is the same as the VCR test set.

1. You can download VCR-OOD from : 
    * `https://drive.google.com/drive/folders/1QgytMcphialrTGJKYVGlu6E1rRckjEin?usp=drive_link`
   VCR-OOD-QA:
    train 
    * `https://drive.google.com/file/d/1GIwds-UK3ZCJu3PDdKzW3qzh05bnvoJ4/view?usp=drive_link`
    val 
    * `https://drive.google.com/file/d/1VGrSvn_OdmBEgDbSvcwBJUU0xJ0SHY6k/view?usp=drive_link`
    test 
    *`https://drive.google.com/file/d/1_fIlam7kPwEQUa9F_A5sLnHC8xkU_LCO/view?usp=drive_link`
   VCR-OOD-VA: 
    train 
    * `https://drive.google.com/file/d/1GIwds-UK3ZCJu3PDdKzW3qzh05bnvoJ4/view?usp=drive_link`
    val  
    * `https://drive.google.com/file/d/11SKVkAJ6eTeGRAMySdeNffzSCHwzNFOf/view?usp=drive_link`
    test 
    *`https://drive.google.com/file/d/1_fIlam7kPwEQUa9F_A5sLnHC8xkU_LCO/view?usp=drive_link`

2. Pre-trained attribute capturing visual representations are in :
    * `https://drive.google.com/drive/folders/1QgytMcphialrTGJKYVGlu6E1rRckjEin?usp=drive_link`

###################################################VCR#####################################################################################################################

Obtain the dataset by visiting [visualcommonsense.com/download.html](https://visualcommonsense.com/download.html). 
 - Extract the images somewhere. I put them in a different directory, `/home/vcr1/vcr1images` and added a symlink in this (`data`): `ln -s /home/vcr1/vcr1images`
 - Put `train.jsonl`, `val.jsonl`, and `test.jsonl` in here (`data`).
 
You can also put the dataset somewhere else, you'll just need to update `config.py` (in the main directory) accordingly.
```
unzip vcr1annots.zip
```

# Precomputed representations
1. Running CMR_sGCN_attr requires computed bert representations in this folder. Warning: these files are quite large. You can download them from :
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_train.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_train.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_val.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_val.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_test.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_test.h5`

2. Pre-trained attribute capturing visual representations are generated using code in [Bottom Up Attention](https://github.com/peteanderson80/bottom-up-attention), released by paper [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering, Peter Anderson et al., 2018](https://arxiv.org/abs/1707.07998)

