# Microsoft AI Challenge India 2018
https://competitions.codalab.org/competitions/20616

## Problem Statment:
Given a user query and candidate passages corresponding to each, the task is to mark the most relevant passage which contains the answer to the user query.

In the data, there are 10 passages for each query out of which only one passage is correct.  
Therefore, only one passage is marked as label 1 and all other passages for that query are marked as label 0.  
Your goal is to rank the passages by scoring them such that the actual correct passage gets as high score as possible.

## DataSet Description:
We have got two datasets:
1. train set (data.tsv): This is the labelled data the participant will use for building models and validating them.
2. eval1 set (eval1_unlabelled.tsv): This is the unlabelled data which the participants shall use to run their models on and submit predictions for

The columns in data.tsv are -   
query_id, query, passage_text, label, passage_id  

The eval sets have same columns except they don't contain the label columns.

Description of the columns:  
**query_id:** this is an integer denoting the id of the query. For submissions, you should ensure that you give correct passage scores for the appropriate query.  
**query:** a string representing the user query or question. each query has 10 passages. So there will be 10 rows having same query but different passages.  
**passage_text:** it is a particular passage.  
**label:** this is a label having either 0 or 1 which implies whether the passage is relevant for the query. Only one passage will be relevant to the query.   
**passage_id:** This is an integer id for the passage (ranging from 0 to 9) which denotes the order of the passage in the original data.   
For example, if passage_id is 4, then you are looking at the 5th passage. While submitting scores, you should ensure that your scores are in the same order from 0 to 9.  

For sake of illustration, we show a sample row in train set (data.tsv) below:

## Step-wise Procedure
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")
