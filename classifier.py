import pickle


class Classifier:
    def __init__(self):
        pass

    def deserialize(self):
        pass

    def classify(self, text):
        deserialized = self.deserialize()
        model = deserialized[0]
        tfidf = deserialized[1]
        id_map = deserialized[2]

        text_features = tfidf.transform([text])
        category = id_map[model.predict(text_features)[0]]

        return category


class GeneralClassifier(Classifier):

    # overriding abstract method
    def deserialize(self):
        with open("gen_clf.pickle", "rb") as gen_clf_file:
            gen_clf = pickle.load(gen_clf_file)

        with open("gen_tfidf.pickle", "rb") as gen_tfidf_file:
            gen_tfidf = pickle.load(gen_tfidf_file)

        with open("gen_id_map.pickle", "rb") as gen_id_map_file:
            gen_id_map = pickle.load(gen_id_map_file)

        return [gen_clf, gen_tfidf, gen_id_map]

    def classify(self, text):
        return super().classify(text)


class ConfidentialClassifier(Classifier):

    # overriding abstract method
    def deserialize(self):
        with open("conf_clf.pickle", "rb") as conf_clf_file:
            conf_clf = pickle.load(conf_clf_file)

        with open("conf_tfidf.pickle", "rb") as conf_tfidf_file:
            conf_tfidf = pickle.load(conf_tfidf_file)

        with open("conf_id_map.pickle", "rb") as conf_id_map_file:
            conf_id_map = pickle.load(conf_id_map_file)

        return [conf_clf, conf_tfidf, conf_id_map]

    def classify(self, text):
        return super().classify(text)


'''
with open("gen_id_map.pickle", "rb") as gen_id_map_file:
    gen_id_map = pickle.load(gen_id_map_file)

with open("gen_tfidf.pickle", "rb") as gen_tfidf_file:
    gen_tfidf = pickle.load(gen_tfidf_file)

model_file = open("gen_clf.pickle","rb")
model = pickle.load(model_file)
model_file.close()

texts = "An amended draft bill on the regulation of Islamic institutes including Madrasas will be presented to the Cabinet after consultations with the Attorney General’s Department and the Muslim Religious Affairs Ministry, Prime Minister Ranil Wickremesinghe said. The Prime Minister said the country does not need Sharia universities and degree awarding institutions should be open to all under the Universities Act."
text_features = gen_tfidf.transform([texts])
predictions = model.predict(text_features)
print(predictions)
print(gen_id_map[predictions[0]])'''
