test_df = pd.read_csv("/data/downloaded/eval2_unlabelled.tsv",sep = '\t', header = None)
test_df = test_df.rename(columns = {0:'index',3:4})
test_df.to_csv('data/test_eval2.tsv',sep = '\t',index = False)

query_series = test_df.loc[:,'index']
query_unique = query_series.unique()
query_unique_df = pd.Dataframe(query_unique)
query_unique_df.to_csv('data/query_ids/test.ids',header = None,index = False)

