# deep-learning-challenge
##Overview of the Analysis
The purpose of this analysis is to create an algorithm that assists a non-profit foundation in identifying the most suitable applicants for funding their venture projects. This algorithm will utilize machine learning and deep learning neural networks to predict the likelihood of an applicant’s success if funded. The model will be developed using metadata from over 34,000 organizations that have previously received funding from the foundation.

##Results: 
Data Preprocessing
The following is a snapshot of the original dataframe:
What variable(s) are the target(s) for your model?
•	IS_SUCCESSFUL is the target column

##What variable(s) are the features for your model?
•	APPLICATION_TYPE, AFFILIATION,CLASSIFICATION,USE_CASE,ORGANIZATION,STATUS,	INCOME_AMT,	SPECIAL_CONSIDERATIONS, ASK_AMT. These features were used in the initial model. 
What variable(s) should be removed from the input data because they are neither targets nor features?
•	EIN', and 'NAME' were removed. 

##Compiling, Training, and Evaluating the Model
How many neurons, layers, and activation functions did you select for your neural network model, and why?
•	Initial Model:
o	Two hidden layers and one output layer. 
o	First hidden layer-8 neurons with Rectified linear unit (ReLU) activation function
o	Second hidden layer-5 neurons with ReLU activation function
o	Output layer- 1 neuron with Sigmoid activation function

•	Optimised Model 1:
o	Two hidden layers and one output layer. 
o	First hidden layer-30 neurons with Rectified linear unit (ReLU) activation function
o	Second hidden layer-20 neurons with ReLU
o	Third hidden layer- 10 neurons with ReLU
o	Output layer- 1 neuron with Sigmoid activation function

Were you able to achieve the target model performance?
A target predictive accuracy higher than 75% was desirable but was not achieved. 
The following steps were taken to increase model performance in a separate notebook file `AlphabetSoupCharity_Optimisation-1.ipynb`: 
•	The column ‘SPECIAL_CONSIDERATIONS ‘was dropped from the training dataset as it is suspected of causing confusion in the model. 
•	The cut-off values for ‘APPLICATION_TYPE’ and ‘CLASSIFICATION’ columns were decreased to 100 to include more categories of values for training. 
•	A third hidden layer added with neurons increased to 10,20,10 to each of the three hidden layers respectively. 
•	Increased test size split to 0.25 from 0.20 for more accurate estimate of the model’s performance on unseen data
•	Batch size was reduced to 64 ton improve generalisation of the model. 
•	Epocs were increased to 150 to give the model more opportunities to learn from the training data. 

##Summary
The initial model produced a predictive accuracy score of 72.4 percent. This was lower than the desired score of 75 percent and above. After optimisation steps were applied the model’s accuracy score improved to 73.3 percent. It was only a slight increase and still did not meet the desired accuracy rate of 75 percent or more. A third attempt was also made with tweaks on the various steps highlighted above but the accuracy score still remains very close to the result in the first optimisation attempt. 

##Recommendation
To enhance the predictive accuracy of the classification problem, an ensemble learning approach such as Random Forest, Gradient Boosting, or XGBoost could be highly effective. These methods combine the predictions of multiple models to leverage their strengths and mitigate individual weaknesses, resulting in improved overall performance. For instance, Random Forests generate multiple decision trees using different subsets of the data and features, then aggregate their predictions to reduce overfitting and enhance generalization. Similarly, Gradient Boosting and XGBoost build models sequentially, where each new model corrects the errors of the previous ones, focusing on difficult-to-predict instances. These ensemble techniques are well-suited for classification tasks and can handle diverse data types and distributions, potentially leading to better results than a single deep learning model.
