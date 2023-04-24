import pickle

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report


def main():
    model = load_model("model_dumps/model.pkl")
    test_data = load_test_data("data/preprocessed_data.csv")

    y = test_data['Delay']
    X = test_data.drop(['Delay'], axis=1)
    X = X.drop(['id'], axis=1)
    
    page = st.sidebar.selectbox(
        "Выберите страницу",
        ["Описание задачи и данных", "Запрос к модели"]
    )

    if page == "Описание задачи и данных":
        st.title("Описание задачи и данных")
        st.write("Выберите страницу слева")

        st.header("Описание задачи")
        st.markdown("""
        Набор данных содержит информацию о воздушных перелетах американских авиакомпаний. 
        Цель этой модели - прогнозирование задержки рейсовна на основе определенных параметров, доступных в наборе данных. 
        """)

        st.header("Описание данных")
        st.markdown("""Предоставленные данные:
        * Flight - Номер рейса,
        * DayOfWeek - день недели,
        * Time - время полета,
        * Length - длинна перелета,
        * Delay - задержка рейса,
        все признаки - вещественные
* К порядковым признакам относится день недели, данный признак принимает значения 0-7, по дням недели соответсвенно.
* К бинарным признакам относятся:
* задержка рейса принимает значения 0 и 1, где значение 0 означает отсутсвие задержки рейса, а значение 1 – наличие задержки;
* К вещественным признакам относятся:
* номер рейса,
* время полета,
* длинна перелета.""")

    elif page == "Запрос к модели":
        st.title("Запрос к модели")
        st.write("Выберите страницу слева")
        request = st.selectbox(
            "Выберите запрос",
            ["Метрики", "Первые 10 предсказанных значений", "Сделать прогноз"]
        )

        if request == "Метрики":
            st.header("Метрики")
            y_pred = model.predict(X)
            st.write(classification_report(y, y_pred))
            st.write(confusion_matrix(y, y_pred))
            #st.write(f"{rmse}")
        elif request == "Первые 10 предсказанных значений":
            st.header("Первые 10 предсказанных значений")
            y_pred = model.predict(X.iloc[:10,:])
            for item in y_pred:
                st.write(f"{item:.2f}")
        elif request == "Сделать прогноз":
            st.header("Сделать прогноз")

            Flight = st.number_input("Flight", 0., 10000.)

            DayOfWeek = st.number_input("DayOfWeek", 0, 8)

            Time = st.number_input("Time", 0, 10000)

            Length = st.number_input("Length", 0., 10000.)           

            if st.button('Предсказать'):
                data = [Flight,	DayOfWeek,	Time,	Length]
                data = np.array(data).reshape((1, -1))
                pred = model.predict(data)
                st.write(f"Предсказанное значение: {pred[0]:.2f}")
            else:
                pass



@st.cache_data
def load_model(path_to_file):
    with open(path_to_file, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


@st.cache_data
def load_test_data(path_to_file):
    df = pd.read_csv(path_to_file, sep=";")
    return df


if __name__ == "__main__":
    main()