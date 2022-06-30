
## Этот репозиторий содержит код для задачи "Sentiment analysis", или анализ настроений.

<br>

Исходная задача: Обучение классификатора, определящего вероятность того, что комментарий о банке является негативным.

### Данные
Датасет найден на [kaggle](https://www.kaggle.com/datasets/kicelevalexey/bank-app-reviews-rus) и содержит 7,5к отзывов про банковское приложение.

Из признаков:
* сам отзыв (русский текст);
* заголовок, дата публикации;
* имя пользователя;
* ответ на отзыв со стороны банка;
* флаг: правилось ли сообщение пользователем.

GT в данном случае является рейтинг, который пользователь выставляет приложению. Из числового формата отзыва (1 - 5 звезд) был получен таргет, используя бинаризацию: отзыв признается негативным, если рейтинг < 2.

<br>

### Начало работы

1. conda create --name test python=3.7
2. pip3 install -r requirements.txt
3. Для тренировки: python [train.py](train.py)
   
   Для тестирования: python [inference.py](inference.py) --review 'Очень классно'

<br>

### Обучение
Файл [train.py](train.py) содержит обучение лучшей модели из найденных.
Файл 'sentiment analysis.ipynb' содержит исследования и отбор лучшей модели.
[inference.py](inference.py) содержит код для прогноза тональности, написанного на клавиатуре отзыва.

Для нахождения оптимальных гиперпараметров модели использовалась кросс валидация.

В качестве основы использовались модели RuBert (top-7) и Sbert (top-2). [Бенчмарк](https://huggingface.co/sismetanin/sbert-ru-sentiment-rusentiment).
<br>

### Метрика
В качестве целевой метрики использовался f1. Мотивация в бизнесовом моменте о важности найти все единички, или найти все единички точно. Зависит от компании/ситуации.

<br>

### Бенчмарк


<table>
    <thead align="center">
        <tr>
            <th rowspan="2"><span style="font-weight:bold">Модель</</th>
            <th colspan="1">Используемые фичи</th>
            <th colspan='1'>Кол-во признаков</th>
            <th colspan="1">F1</th>
        </tr
    </thead>
    <tbody align="center">
        <tr>
            <td rowspan="1">RuBert</td>
            <td>Отзыв</td>
            <td>3</td>
            <td>0.674</td>
        </tr>
        <tr>
            <td rowspan="1">RuBert</td>
            <td>Отзыв + Заголовок</td>
            <td>6</td>
            <td>0.723</td>
        </tr>
        <tr>
            <td rowspan="1">RuBert + LogReg</td>
            <td>Отзыв + Заголовок</td>
            <td>6</td>
            <td>0.825</td>
        </tr>
        <tr>
            <td rowspan="1">RuBert + SVC</td>
            <td>Отзыв + Заголовок</td>
            <td>6</td>
            <td>0.825</td>
        </tr>
        <tr>
            <td rowspan="1">RuBert + KNN</td>
            <td>Отзыв + Заголовок</td>
            <td>6</td>
            <td>0.826</td>
        </tr>
        <tr>
            <td rowspan="1">RuBert + Ensemble</td>
            <td>Отзыв + Заголовок</td>
            <td>6</td>
            <td>0.824</td>
        </tr>
        <tr>
            <td rowspan="1">RuBert + LogReg</td>
            <td>Отзыв + Заголовок + sin/cos/onehot date</td>
            <td>34</td>
            <td>0.826</td>
        </tr>
        <tr>
            <td rowspan="1">SBert + LogReg</td>
            <td>Отзыв</td>
            <td>1024</td>
            <td><b>0.829<b></td>
        </tr>
        <tr>
            <td rowspan="1">SBert + KNN</td>
            <td>Отзыв</td>
            <td>1024</td>
            <td>0.826</td>
        </tr>
    </tbody>
</table>

<br>

Итоговая модель [здесь](logreg.sav). Является надстройкой над фичами Sbert и находится в git lfs.