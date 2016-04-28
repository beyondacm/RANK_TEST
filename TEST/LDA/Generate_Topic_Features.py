import graphlab
import pandas as pd

topic_model = graphlab.load_model('/Users/gaozhipeng/ML/RANK_TEST/TEST/LDA/Topic_Model/')

predict_docs = graphlab.SFrame.read_csv('/Users/gaozhipeng/ML/RANK_TEST/TEST/LDA/TEST_LDA_SOURCE.txt', header=False)
predict_docs = graphlab.text_analytics.count_words(predict_docs['X1'])
predict_docs = predict_docs.dict_trim_by_keys(graphlab.text_analytics.stopwords(), exclude=True)

pred = topic_model.predict(predict_docs, output_type='probability')
pred.save('/Users/gaozhipeng/ML/RANK_TEST/TEST/LDA/Raw_Topic_Features.csv', format='csv')

tf = pd.read_table('/Users/gaozhipeng/ML/RANK_TEST/TEST/LDA/Raw_Topic_Features.csv', sep = ' |"|\[|\]', header=None)
tf = tf.drop(tf.columns[[0, 1, 52, 53]], axis=1)
tf.to_csv('/Users/gaozhipeng/ML/RANK_TEST/TEST/LDA/Raw_Topic_Features.csv', encoding='utf-8', index = False, header=None)
