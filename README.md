# Titanic-Machine-Learning
## CONTEXT
The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, 
during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with 
an iceberg. 
Unfortunately, there were not enough lifeboats for everyone onboard, resulting in the death of 1502 
out of 2224 passengers and crew. While there was some element of luck involved in surviving, it 
seems some groups of people were more likely to survive than others.
In this challenge, we ask you to build a predictive model that answers the question: “what sorts of 
people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic 
class, …).
## DATA
The data has been split into two groups:
• training set (train.csv)
• test set (test.csv).
The training set should be used to build your machine learning models. For the training set, we 
provide the outcome (also known as the “ground truth”) for each passenger. Your model will be 
based on “features” like passengers’ gender and class. You can also use feature engineering to 
create new features.
The test set should be used to see how well your model performs on unseen data. For the test set, 
we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For 
each passenger in the test set, use the model you trained to predict whether or not they survived 
the sinking of the Titanic.
## TO DO 
1. Make an EDA.
2. Preprocess the data with the function « ColumnTransformer».
3. K-NN Classifier with GridSearchCV
4. Logistic regression classifier with GridSearchCV
5. Random Forest classifier With GridSearchCV
6. Implement a web application using streamlit where the user can:
- Select the ML model to use.
- Hyperparameters setting.
- Run the classification with classify button!
- Finally, the application prints a confusion matrix
