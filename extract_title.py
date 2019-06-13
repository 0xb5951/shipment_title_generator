import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ベクトル化する文字列
sample = np.array(['Apple computer of the apple mark',
                   'linux computer', 'windows computer'])

# TfidfVectorizer
vec_tfidf = TfidfVectorizer()

# ベクトル化
X = vec_tfidf.fit_transform(sample)

print('Vocabulary size: {}'.format(len(vec_tfidf.vocabulary_)))
print('Vocabulary content: {}'.format(vec_tfidf.vocabulary_))




def predict(self, body, body_plain=None):
        '''
        @param str body: 分かち書きしたものを渡すこと
        @return dict
        '''

        if body_plain:
            # ビジネスルールの適用
            # msgの場合は分かち書きする前に適用する
            bf = BizFilter()
            res_bf = bf.work(body_plain)

            if res_bf:
                return {
                    'predict': 1,
                    'score': app.config['SCORE_THRESHOLD_WORK_SPAM'] + 1.0,
                    'vocabulary': res_bf['keyword']
                }

        # TFIDFはiterableな値しか受けつけないので、リストで渡す
        tfidf = app.config['vect'].transform([body])
        lsa_reduced = app.config['lsa'].transform(tfidf)
        predict = app.config['clf'].predict(lsa_reduced)

        score = self._get_score(app.config['clf'], lsa_reduced)

        vocabulary = self._get_vocabulary(app.config['vect'], tfidf)
