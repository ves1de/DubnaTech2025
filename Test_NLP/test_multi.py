import spacy


text = "Я люблю взаимодействовать с людьми и помогать людям"


nlp = spacy.load("prof_model/ru_multiclass_model44%")
doc = nlp(text)
print(doc.cats)

nlp = spacy.load("prof_model/ru_multiclass_model45%")
doc = nlp(text)
print(doc.cats)

nlp = spacy.load("prof_model/ru_multiclass_model 29%")
doc = nlp(text)
print(doc.cats)


text = "В школе мы пробовали создавать сайты. Мне так понравилось,что это стало моим главным увлечением. Хотел бы и дальше развиваться в этом направлении и попробовать что-то похожее"
nlp = spacy.load("prof_model/ru_multiclass_model44%")
doc = nlp(text)
print(doc.cats)

nlp = spacy.load("prof_model/ru_multiclass_model45%")
doc = nlp(text)
print(doc.cats)

nlp = spacy.load("prof_model/ru_multiclass_model 29%")
doc = nlp(text)
print(doc.cats)

text = "Я ходила в художественную школу 8 лет. Ходить туда -для меня большое удовольствие. Мои любимые предметы в школе это информатика и математика. Хотя в них я погружена не так глубоко как в рисование. Хотела бы попробовать себя в диджитале."
nlp = spacy.load("prof_model/ru_multiclass_model45%")
doc = nlp(text)
print(doc.cats)

nlp = spacy.load("prof_model/ru_multiclass_model44%")
doc = nlp(text)
print(doc.cats)

nlp = spacy.load("prof_model/ru_multiclass_model 29%")
doc = nlp(text)
print(doc.cats)