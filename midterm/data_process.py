# build a function to clean dataframe
def clean_df(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype == "category" or dataframe[i].dtype == "object":
            # categorical datatype fillna with mode
            num = dataframe[i].mode()[0]
            dataframe[i].fillna(num, inplace=True)

        elif len(set(dataframe[i])) < 0.05 * len(dataframe[i]):
            # categorical datatype fillna with mode
            num = dataframe[i].mode()[0]
            dataframe[i].fillna(num, inplace=True)
            # dataframe[i] = labelencoder.fit_transform(dataframe[i])
        else:
            # numerical datatype fillna with mean
            num = dataframe[i].mean()
            dataframe[i].fillna(num, inplace=True)
    return dataframe


# Build a function to Determine if response is continuous or boolean (don't worry about >2 category responses)
def response_con_bool(response_list):
    if len(set(response_list)) == 2:
        return "boolean"
    else:
        return "continous"


# Determine if the predictor is cat/cont
def cont_bool(predictor_list):
    if predictor_list.dtype == "category" or predictor_list.dtype == "object":
        return "categorical"
    elif len(set(predictor_list)) == 2:
        return "categorical"
    else:
        return "continous"
