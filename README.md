# Financial_Data_Modeling

This is a project on data retrieved from www.kaggle.com/wordsforthewise/lending-club.
The goal of this project is to create a model which can predict if a user can or cannot repay a loan.
After ETL process the data is normalized and loaded on a simple sequential front-feeding neural network .
The model is trained with a rectified linear unit activation function and using ADAM as an optimizer.






 precision    recall  f1-score   support

           0       0.99      0.44      0.61     15658
           1       0.88      1.00      0.93     63386

    accuracy                           0.89     79044
   macro avg       0.93      0.72      0.77     79044
weighted avg       0.90      0.89      0.87     79044
