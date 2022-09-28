# KenLM Language Model

Language models boost the performance of ASR models. LM are built with text from our target corpus and the ASR pipeline uses this model to find probabilities of different sequence of words and pick a more probable sentence. 

KenLM is a classic language model. It implements interpolated modified Kneser Ney Smoothing. KenLM is a robust open source LM and we decided to train LMs with transcripts available.

Test input file: transcripts2022_02_06.csv

## Scripts

### Preprocessing Scripts
Text preprocessing scripts specific to the transcript files created with protocols such as transcripts2022_02_06.csv

Different approaches to preprocessing was selected based on the transcripts available and the obvious errors/features in them.
1.	process_bpc_tscript.py – simple preprocessing the transcript corpus like removing brackets, punctuations, nonascii characters. Script should be self explanatory with lots of comments
2.	process_keep_sqrbrkt_text_bpc_tscript.py – text within [ ] are kept. brackets alone deleted
3.	process_keepx_bpc_tscript - <X> is retained 
4.	process_superclean_bpc_tscript.py  - deletes full sentences if it so much has an <X> or text within [ ]


### Running Kenlm:

klm_cv_10fold_v2.py
Does it all.
1.	Takes the preprocessed input csv file
2.	Builds the kenlm ARPA / binary file using kenLM engine installed in your env
3.	Uses KFold cross validation to split corpus into test, train and build
4.	Estimates the log probabilities and perplexity scores of sentences in test
5.	Output is an excel file
    a.	Means of log probabilities and perplexity score for each KFold run
    b.	Each Kfold run documented in separate tabs with score of every individual sentence



