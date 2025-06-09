# Log Analysis with BERT and Naive Bayes

Цей проєкт демонструє аналіз системних логів з використанням BERT для векторизації та Naive Bayes для класифікації. Він дозволяє автоматично кластеризувати та класифікувати системні логи, що допомагає у виявленні аномалій та аналізі поведінки системи.

## Швидкий старт

1. Встановіть залежності:
```bash
pip install -r requirements.txt
```

2. Запустіть аналіз:
```bash
python naive_bayes_classifier.py
```

## Структура проєкту

- `naive_bayes_classifier.py` - основний класифікатор
- `bert_vectorization_pipeline.py` - векторизація логів за допомогою BERT
- `log_parsing_pipeline.py` - парсинг та обробка логів
- `cluster_visualization.py` - візуалізація кластерів

## Набір даних

Використовується набір даних з [Zookeep_2k.log з LogHub](https://github.com/logpai/loghub/tree/master/Zookeeper):

Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. [Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics](https://arxiv.org/abs/2008.06448). IEEE International Symposium on Software Reliability Engineering (ISSRE), 2023. 