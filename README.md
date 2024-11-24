# Data Challenge - SNCF-Transilien

This repository contains the work and code for the **SNCF-Transilien Data Challenge**. The challenge's goal is to predict the number of daily validations (ticket validations) at various stations of the SNCF-Transilien network, focusing on improving prediction accuracy for future validation volumes. This prediction will help optimize the services and operations of the trains running in the Île-de-France region.

## Objective
The main objective of this challenge was to build a machine learning model capable of forecasting the daily number of ticket validations at 448 stations in the SNCF-Transilien network. We used historical validation data from January 2015 to December 2022 to train the model and predict future validations for the first half of 2023.

## Dataset
The dataset consists of daily validation data for the SNCF-Transilien network stations. It includes several features such as:
- **Date**: The date when validations were recorded.
- **Station**: The unique identifier for each station.
- **Job**: A binary indicator (1 or 0) indicating whether the day is a weekday.
- **Holiday**: A binary indicator (1 or 0) for public holidays.
- **School Vacation**: A binary indicator (1 or 0) for school holidays.

We received the following files:
- **train_x.csv**: Contains feature data (dates, station, job, holiday, and school vacation indicators).
- **train_y.csv**: Contains the target variable (number of validations) along with the corresponding index, which is a combination of the station and the date.
- **test_x.csv**: Contains feature data for the prediction period (January 2023 to June 2023).

## Approach

### Data Preprocessing
- We cleaned and preprocessed the data by handling missing values and converting categorical features (like station names) into a more efficient format.
- We extracted additional features from the date column, such as day of the week, month, and year, to improve model accuracy.
- We also created lag features to capture the temporal dependencies in the data, which helped the model learn from past validation patterns.

### Models
- We experimented with various machine learning models, including **Random Forest**, **Prophet**, and **Poisson Regression**.
- Ultimately, we focused on boosting algorithms, particularly **LightGBM**, which provided the best performance for this task.
- For model optimization, we performed **Grid Search** to find the best hyperparameters, considering the computational limits of our system.

### Results
- The **LightGBM** model outperformed other models in terms of prediction accuracy and efficiency.
- By adding temporal features (lags) and fine-tuning the model using grid search, we were able to achieve a significant improvement in model performance.

## Structure
The repository is organized as follows:
- `data/`: Folder containing the dataset files.
- `ml_pred_lightGBM.ipynb`: Jupyter notebook with the final model implementation and predictions.
- `other models`: Folder containing other models we experimented with. They are not really finished, but they show the process we went through.
- `README.md`: This file.
- `LICENSE`: The license for the repository.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- SNCF-Transilien for providing the dataset.
- Open source libraries like **LightGBM** and **scikit-learn** for their efficient implementations of machine learning models.
