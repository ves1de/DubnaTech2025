import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from spacy.training.example import Example
from spacy.util import minibatch

# Загружаем данные
df = pd.read_csv('DataSet/prof1000.csv', on_bad_lines='skip', encoding='utf-8')

# Удаляем строки без текста или метки
df = df.dropna(subset=['text', 'sentiment'])

# Оставляем только нужные колонки
df = df[['text', 'sentiment']]

# Делим на обучающую и тестовую выборки
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Получаем список уникальных классов
labels = df['sentiment'].unique().tolist()

# Функция преобразования в формат spaCy
def convert_to_spacy_format(data):
    result = []
    for _, row in data.iterrows():
        cats = {label: False for label in labels}
        cats[row['sentiment']] = True
        result.append((row['text'], {"cats": cats}))
    return result

train_data = convert_to_spacy_format(train_df)
test_data = convert_to_spacy_format(test_df)

# Создаём пустую модель spaCy
nlp = spacy.blank("ru")


# Добавляем pipeline для классификации текста
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat", last=True)

# Регистрируем метки
for label in labels:
    textcat.add_label(label)

# Обучение модели
n_iter = 25
optimizer = nlp.begin_training()

for i in range(n_iter):
    losses = {}
    batches = minibatch(train_data, size=8)
    for batch in batches:
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        nlp.update(examples, sgd=optimizer, losses=losses)
    print(f"Losses at iteration {i}: {losses}")

# Оценка
correct = 0
total = 0

for text, annotations in test_data:
    doc = nlp(text)
    predicted_label = max(doc.cats, key=doc.cats.get)
    true_label = max(annotations['cats'], key=annotations['cats'].get)
    if predicted_label == true_label:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Обработано: {total}, Верно: {correct}")
print(f"Точность модели: {accuracy:.2f}")

# Сохраняем модель
nlp.to_disk("ru_multiclass_model")
print("Модель успешно сохранена!")

print(df['sentiment'].value_counts())

from sklearn.metrics import classification_report

y_true = []
y_pred = []

for text, annotations in test_data:
    doc = nlp(text)
    predicted_label = max(doc.cats, key=doc.cats.get)
    true_label = max(annotations['cats'], key=annotations['cats'].get)
    y_true.append(true_label)
    y_pred.append(predicted_label)

print(classification_report(y_true, y_pred))
