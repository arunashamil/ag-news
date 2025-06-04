# Классификация новостей по их тематике на основе датасета AG News

## Постановка задачи

Задача состоит в распределении новостных статей по четырем основным категориям
(классам): “World”, “Sport”, “Business”, “Science/Technology”. Классификация
происходит на основе названия и краткого описания статьи. Разбиение по
категориям делает поиск интересующих новостей удобнее и быстрее.

Проект также полезен для образовательных целей: на простой задаче классификации
можно разобраться в том, как предобрабатываются тексты, отточить принципы работы
рекуррентных сетей, а также увидеть, как выбор модели и ее параметров влияет на
качество предсказания.

## Формат входных и выходных данных

Входные данные:

Датасет AG News, используемый при обучении, имеет следующую структуру: всего
имеется 120000 объектов (по 30000 на каждую новостную категорию) для тренировки
и 7600 (по 1900 на каждую тему) для теста. У каждого объекта выборки по 3 поля:
индекс класса, название статьи, ее описание.

Выходные данные:

Модель предсказывает тему новостной статьи как один из вышеперечисленных
классов.

## Метрики

В задаче используется метрика accuracy. Поскольку датасет сбалансированный, а
тематики статей независимы друг от друга, accuracy хорошо подходит.

## Валидация

Поскольку в исходном датасете есть только train и test части, “возьмем” от
train-а 5% объектов с помощью функции
[random_split из torch.utils.data.dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split).
Во время валидации мы смотрим, как падает loss и растет accuracy. Для
визуализации используется система логирования [WandB](https://docs.wandb.ai/).

## Данные

### Данные

В задаче дла train-а и test-а модели используется датасет новостей AG News
(https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset),
[ссылка на диск](https://drive.google.com/file/d/1M84gtv_8mFSkLh3SCIKcy10fr7BKP73l/view?usp=sharing).

Датасет сбалансированный, но отсутствует выделенная валидационная выборка, что
легко решается.

### Потенциальные проблемы:

Проблемы, которые могут возникнуть, вполне стандартные:

1. Датасет не очень большой. Если на вход подать статью, не похожую на
   тренировочные (например, научную статью на тему, которая в тренировочной
   выборке никак не освещалась), то при ее токенизации будет много unknown
   токенов и она не сможет использовать важную информацию из них при
   классификации;
2. Если на вход подать статью на экзотическую тему (например, кулинария), то
   ответ модели заведомо ошибочный.

## Моделирование

### Бейзлайн

1. Модель должна справляться лучше чем модель, выдающая случайный класс в
   качестве предсказания;

2. Для сравнения в качестве линейного слоя можно взять не только
   [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html), но и
   [RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html).

### Основная модель

За основу данного проекта была взята
[модель](https://github.com/axiom2018/AG-News-Classification/blob/main/AG%20News%203%20Models%201%20used%20with%20Optuna.ipynb).

0. Предобработка данных

Данные из .csv файлов конвертируются в pd.DataFrame, затем посредством класса

```python
class TextDataset(Dataset):
    """Dataset class for torch dataloaders"""

    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return torch.tensor(self.sentences[index], dtype=torch.long), torch.tensor(
            self.labels[index], dtype=torch.long
        )
```

мы преобразуем их в датасет (только после этого шага можно применить метод
random_split). По всем словам, имеющимся в тренировочном датасете, мы составляем
словарь и токенизируем тексты. Для дальнейшего разбиения на батчи дополняем
короткие строки токеном <pad>. При предобработке тестовых данных неизвестные
слова токенизируются как <unk>.

В данном проекте активно используется библиотека
[PyTorch](https://pytorch.org/), поэтому после предобработки подаем наши данные
в
[torch.nn.DataLoader](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader),
это упрощает написание кода и ускоряет загрузку данных;

1. Модель:

```python
self.embedding = torch.nn.Embedding(vocab_size, input_size)
self.lstm = torch.nn.LSTM(
           input_size=input_size,
           hidden_size=hidden_size,
           num_layers=num_layers,
           batch_first=True,
           dropout=dropout if num_layers > 1 else 0,
       )

self.dropout = torch.nn.Dropout(dropout)
self.fc = torch.nn.Linear(hidden_size, num_classes)
```

Аналогичную структуру имеет модель с линейным слоем RNN;

2. Оптимизатор

В качестве оптимизатора в данной задаче используется
[Adam][https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html] с
LR scheduler-ом
[ExponentialLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html);

3. Минимизируем функцию потерь
   [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html);

4. Модель обучается 10 эпох.

## Внедрение

Модель может использоваться как пакет для классификации новостных статей. В
проекте осуществляется перевод обученной модели в формат .onnx для дальнейшей
работы над моделью как продуктом.

# Работа с проектом

## Setup

Клонирование репозитория

```
git clone https://github.com/arunashamil/ag-news.git
```

После клонирования репозитория перейдите в его папку

```
cd ag-news
```

Установите Poetry для управления зависимостями

```
pip install poetry==2.1.2
```

Установка зависимостей проекта

```
poetry install
```

Установка хуков для pre-commit

```
poetry run pre-commit install
```

Запустите проверки

```
poetry run pre-commit run --all-files
```

## Структура папок

```
ag_news
\----data_load
|        ---download_data.py
|
\----infer
|        ---infer.py
|
\----modules
|        ---compile_to_onnx.py
|
\----train
|        ---train.py
```

## Train

Активируйте окружение

```
poetry env activate
```

Для отслеживания метрик в ходе обучения и валидации необходимо авторизоваться в
WandB. По предложенной ссылке перейдите в личный кабинет, скопируйте и введите
API-ключ в командную

```
poetry run wandb login --relogin
```

Скачайте данные для обучения и тестирования модели

```
cd ag_news/data_load

poetry run python download_data.py
```

Запустите модель обучаться

```
cd ../train

poetry run python train.py
```

## Infer

После того, как модель обучится, запустите систему предсказаний на тестовых
данных ag_news_test.csv с помощью модели, сохраненной во время обучения в папке
models в файле .ckpt.

Название вашего файла может отличаться. Пример запуска:

```
cd ../infer

poetry run python infer.py ag_news_test.csv epoch=09-val_loss=0.3778.ckpt
```

Обученную модель можно перевести из .ckpt в формат .onnx. Не забудьте поменять
название файла:

```
cd ../modules

poetry run python compile_to_onnx.py epoch=09-val_loss=0.3778.ckpt
```
