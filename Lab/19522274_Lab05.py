import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import streamlit as st
import io
#---------------------------#

#--Model--#


def model_scores(data, input_ft, split_size, score_type):
    if len(input_ft) != 1:
        X = data[input_ft].values
    else:
        X = [[value] for value in data[input_ft[0]].values]

    y = data.iloc[:, -1].values
    ##Split train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_size, random_state=0)

    ##Fit LR to training set
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    if score_type == "MAE":
        return cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    else: 
        return cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')



#--Streamlit--#
st.title("Linear Regression with Cross Validation")

uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    data = pd.read_csv(uploaded_file, sep= None)
    st.write(data)
    
    feature_options = data.columns[:-1]
    dft = feature_options[0]

    score_options = ["MAE", "MSE"]

    choosen_ft = st.multiselect(
        "Input feature ",
        options= feature_options,
        default = dft,
    )

    split_size = st.slider("Train/Test split", 0.0, 1.0, 0.8)

    choosen_score = st.radio(
        "Score options ",
        options= score_options,
    )
    lr_model = model_scores(data, choosen_ft, split_size, choosen_score)

    if st.button('Result'):
        st.write(lr_model)
    else:
        st.write()
