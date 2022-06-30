import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import f1_score

from utils import DataFrameSelector, AddTargetEncoding, PreTrainSbertProcessing
import joblib


if __name__ == '__main__':
    
    df = pd.read_csv('sber_app.csv', index_col=0)
    
    pipeline_target = Pipeline([
        ('selector', DataFrameSelector(columns=['rating'])),
        ('encoding', AddTargetEncoding()),
    ])
    
    pipeline_sbert_review = Pipeline([
        ('selector', DataFrameSelector(columns=['review'])),
        ('bert', PreTrainSbertProcessing())
    ])

    df_train, df_test, Y_train, Y_test = train_test_split(
        df[set(df.columns) - set(['rating'])],  # X - без rating
        pipeline_target.fit_transform(df),  # Y - в виде {1 - негативное, 0 - позитивное, или нейтральное}
        test_size=0.25, random_state=42, shuffle=True,
        stratify=df['rating'].values  # стратификация по rating в цифрах, а не в бинарном векторе негативных отзывов. Из первой вытекает вторая
    )
    
    X_train = pipeline_sbert_review.fit_transform(df_train)  # только отзыв
    X_test = pipeline_sbert_review.transform(df_test)
    
    model = LogisticRegression(
        max_iter=10000,
        solver='liblinear',
        random_state=42,
        n_jobs=-1,
        C=0.01,
        class_weight=None,
        penalty='l2',
    )
    
    model.fit(X_train, Y_train)

    pred = model.predict(X_test)
    
    print(f1_score(Y_test, pred))
    
    joblib.dump(model, 'logreg.sav')
