# deep-learning-challenge

## Overview of the Analysis
The purpose of this analysis is to create an algorithm that assists a non-profit foundation in identifying the most suitable applicants for funding their venture projects. This algorithm will utilise machine learning and deep learning neural networks to predict the likelihood of an applicant’s success if funded. The model will be developed using metadata from over 34,000 organizations that have previously received funding from the foundation.

## Results

### Data Preprocessing
The following is a snapshot of the original dataframe which highlights the different variables/features in the dataset:
![original dataframe](./Images/original%20dataset.png)

**Model Target Variable:**
- `IS_SUCCESSFUL`

**Model Features:**
- `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`

**Variables removed from the input data because they are neither targets nor features:**
- `EIN` and `NAME` were removed from the initial model
- `SPECIAL_CONSIDERATIONS` was removed from the optimised model

The following is a snapshot of the updated dataframe from the optimised model:
![updated dataframe](./Images/removed%20columns.png)

### Compiling, Training, and Evaluating the Model

#### Neurons, Layers, and Activation Functions Used

**Initial Model:**
- Two hidden layers and one output layer
  - First hidden layer: 8 neurons with Rectified Linear Unit (ReLU) activation function
  - Second hidden layer: 5 neurons with ReLU activation function
  - Output layer: 1 neuron with Sigmoid activation function
- The above neurons, layers, and activation functions were considered an appropriate benchmark to achieve the desired performance.
- The image below shows the structure of the initial model:

![Neurons and Layers 1](./Images/initial%20model%20layers%20and%20neurons.png)

**Optimized Model 1:**
- Two hidden layers and one output layer
  - First hidden layer: 30 neurons with Rectified Linear Unit (ReLU) activation function
  - Second hidden layer: 20 neurons with ReLU activation function
  - Third hidden layer: 10 neurons with ReLU activation function
  - Output layer: 1 neuron with Sigmoid activation function
- The above tweaks on adding more neurons to the deep learning model and adding an additional layer were done to improve the model’s capacity to learn complex patterns and enhance its overall performance.
- The image below shows the structure of the optimised model:
  
![Neurons and Layers 2](./Images/optimised%20model%20layers%20and%20neurons.png)

#### Target Model Performance Achieved?
A target predictive accuracy higher than 75% was desirable but was not achieved. The following image displays the accuracy score of the initial model:

![Initial Model](./Images/initial%20model%20accuracy.png)

#### Steps Taken to Increase Model Performance
- The column `SPECIAL_CONSIDERATIONS` was dropped from the training dataset as it is suspected of causing confusion in the model.
- The cut-off values for `APPLICATION_TYPE` and `CLASSIFICATION` columns were decreased to 100 to include more categories of values for training.
- A third hidden layer was added with neurons increased to 30, 20, 10 to each of the three hidden layers respectively.
- Increased test size split to 0.25 from 0.20 for a more accurate estimate of the model’s performance on unseen data.
- Batch size was reduced to 64 to improve generalisation of the model.
- Epochs were increased to 150 to give the model more opportunities to learn from the training data.

The above steps were completed in a separate notebook file `AlphabetSoupCharity_Optimisation-1.ipynb`.

The following image shows the accuracy score of the optimised model:

![Optimised Model](./Images/optimised%20model%20accuracy.png)


**Note:** An additional attempt was made to boost optimisation however the accuracy scores remained very close to the first optimisation attempt. This was completed in a separate notebook file `AlphabetSoupCharity_Optimisation-2.ipynb`.

## Summary
The initial model produced a predictive accuracy score of 72.4 percent. This was lower than the desired score of 75 percent and above. After optimisation steps were applied, the model’s accuracy score improved to 73.3 percent. It was only a slight increase and still did not meet the desired accuracy rate of 75 percent or more. A third attempt was also made with tweaks on the various steps highlighted above, but the accuracy score still remains very close to the result in the first optimisation attempt.

## Recommendation
To enhance the predictive accuracy of the classification problem, an ensemble learning approach such as Random Forest, Gradient Boosting, or XGBoost could be highly effective. These methods combine the predictions of multiple models to leverage their strengths and mitigate individual weaknesses, resulting in improved overall performance. For instance, Random Forests generate multiple decision trees using different subsets of the data and features, then aggregate their predictions to reduce overfitting and enhance generalisation. Similarly, Gradient Boosting and XGBoost build models sequentially, where each new model corrects the errors of the previous ones, focusing on difficult-to-predict instances. These ensemble techniques are well-suited for classification tasks and can handle diverse data types and distributions, potentially leading to better results than a single deep learning model.
