from helperFuncs import *
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from helperFuncs import *

mainDataPath = "dbs/avocado.csv"
pd.set_option("display.max_rows", 3)
pd.set_option("display.max_columns", 60)
mainData = None
dummifiedData = None

def main():
    print("start")

    global mainData, dummifiedData

    mainData = pd.read_csv(mainDataPath)

    columns_to_dummify = ["type"]
    dummifiedData = createDummies(mainData, columns_to_dummify)
    dummifiedData = dropObjectColumns(dummifiedData) # if we decide not to dummify something
    dummifiedData.dropna(inplace=True)

    print("info",dummifiedData.info)

    #decide which values to log
    values_to_log = ["AveragePrice", "Date", "Total Volume", "4046", "4225", "4770","Total Bags","Small Bags","Large Bags", "XLarge Bags","year"]
    dummifiedData = logFields(dummifiedData, values_to_log)


## At this point dummified data is loged

    # create the test and train pair:
    x_train, x_test, y_train, y_test, trainingData, testData = createTestTrainPair(dummifiedData, "AveragePrice")
    print(" x_test: ", x_test)
    print("y_test: ", y_test)
    # train linear regression model
    reg = LinearRegression()
    reg.fit(x_train,y_train)
    print("Linear Regression Score: ", reg.score(x_test, y_test))

    # train forest regression model
    forest = RandomForestRegressor()
    forest.fit(x_train, y_train)
    print("Forest Regression Score: ", forest.score(x_test, y_test))   

    # test with test data
    for _ in range(10):
        input("Press Enter for next avocado\n\n")
        testnum = random.randint(1, 500)  # Change the range as needed

        fullTestRow = pd.concat([getRow(y_test, testnum), getRow(x_test, testnum)], axis=1)
        print("fullTestRow", fullTestRow)
        test_row_x = fullTestRow.drop(["AveragePrice"], axis = 1)

        expedTestRow = expFields(fullTestRow, values_to_log)

        print("expedTestRow:: ", expedTestRow)
        # Linear prediction
        prediction = reg.predict(test_row_x)
        prediction_df = pd.DataFrame({"AveragePrice":prediction}, index=test_row_x.index)
        rowWithPrediction = pd.concat([prediction_df, test_row_x], axis = 1)
        rowWithPrediction_exped = expFields(rowWithPrediction, values_to_log)

        print("\nLR Model prediction AveragePrice:\n", rowWithPrediction_exped["AveragePrice"].iloc[0])
        # Forest prediction
        prediction = forest.predict(test_row_x)
        prediction_df = pd.DataFrame({"AveragePrice":prediction}, index=test_row_x.index)
        rowWithPrediction = pd.concat([prediction_df, test_row_x], axis = 1)
        rowWithPrediction_exped = expFields(rowWithPrediction, values_to_log)

        print("\nForest Model prediction AveragePrice:\n", rowWithPrediction_exped["AveragePrice"].iloc[0])

        # Actual value
        print("\nActual AveragePrice:\n", expedTestRow["AveragePrice"].iloc[0])

if __name__ == "__main__":
    main()