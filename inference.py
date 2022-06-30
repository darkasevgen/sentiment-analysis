import argparse
from utils import PreTrainSbertProcessing
import joblib


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Вывод вероятности негативного отзыва')
    parser.add_argument('--review', type=str, required=True)
    
    args = parser.parse_args()
    
    feats = PreTrainSbertProcessing().transform([[args.review]])
    
    model = joblib.load('logreg.sav')

    prob = model.predict_proba(feats)[0][1]  # беру вер-ть негативного отзыва

    print(f'Вероятность негативного отзыва = {round(prob, 3)}')
