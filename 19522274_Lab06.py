from email.policy import default
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, log_loss
import streamlit as st

#---------------------------#

#--Model--#


def model_scores(data, input_ft, split_size, res_option):
    if len(input_ft) != 1:
        X = data[input_ft].values
    else:
        X = [[value] for value in data[input_ft[0]].values]

    y = data.iloc[:, -1].values
    ##Split train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_size, random_state=0)

    ##Fit LR to training set
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    return f1_score(y_test,y_pred) if res_option == "F1_score" else log_loss(y_test, y_pred)



#--Streamlit--#
st.title("Logistic Regression")

uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=None)

    st.write(data)
    
    feature_options = data.columns[:-1]
    dft = feature_options[0]


    choosen_ft = st.multiselect(
        "Input feature ",
        options= feature_options,
        default = dft,
    )

    split_size = st.slider("Train/Test split", 0.0, 1.0, 0.8)

    res_options = ["F1_score", "Log_loss"]

    choosen_res = st.selectbox(
        "Select Score option",
        options = res_options,
    )
    lr_model = model_scores(data, choosen_ft, split_size, choosen_res)

    if st.button('Result'):
        st.write(lr_model)
    else:
        st.write()
