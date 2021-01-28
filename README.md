# SMART-KPE
Code for paper "[Incorporating Multimodal Information in Open-Domain Web Keyphrase Extraction](https://www.aclweb.org/anthology/2020.emnlp-main.140/)"
You can download the data [here](https://victorywys-datasets.s3.us-east-2.amazonaws.com/OpenKP_title_and_snapshot.zip).

Update: 

Since we released the code, we have been working on writing a document and comments for you. In order to make it easy to replicate the result and commpare to previous works, we're  trying to generate checkpoints from all 15 varients according to [BERT-KPE](https://github.com/thunlp/BERT-KPE) upon their version of codes (included in BERT-KPE-BASED folder) and we will release them as soon as possible.

**We provide final checkpoints from BERT-KPE_based [here](https://victorywys-datasets.s3.us-east-2.amazonaws.com/final_checkpoints.zip). Currently we only upload the best model(Roberta2Joint based SMART-KPE, F@3: 0.405) and we'll update more from different varients soon.** You can use the `test.sh` in the script folder to check the results. 

To run the code, make sure you're using Pytorch 1.4.0, otherwise the data parallel part/transformer may not work properly.

If you would like to replicate the best result before we release other checkpoints, you can first try the following steps: 
1. Download the image data and title data. 
2. Add all the title data to the dataset files. In the original jsonl file, each line corresponds to a piece of data and contain 3 domains: `url`, `text` and `VDOM`. In order to use title data, you can add a new domain named `title` and the content is the title string. (We will add a script to help you process it in the near future)
3. Follow instructions of BERT-KPE to proprecess data and run the experiment with scripts provided in this repo. 

Thanks again for your interest of our work!
