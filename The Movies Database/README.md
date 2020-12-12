# The Movies Database

The dataset used in this project is the TMDB 5000 Movie Dataset downloaded from Kaggle (https://www.kaggle.com/tmdb/tmdb-movie-metadata). It consists of 5000 different movies with 20 columns. For the purpose of this project, only 8 columns are used:
  •	Title
  •	Budget
  •	Genre
  •	Popularity
  •	Release Date
  •	Revenue
  •	Runtime
  •	Average Rating

The goal of this project is to do some exploratory data analysis first to extract some insights from the data then build Machine Learning models that should be able to evaluate a movie’s success. A movie’s success is evaluated by Average Rating and Profit (Revenue – Budget). 

A Logistic Regression model is used to classify movies into successful/unsuccessful based on Average Rating: successful movies have an Average Rating >= 6.0. Since there are more 1’s than 0’s, the data is imbalanced and therefor, area under ROC is used to evaluate this model.

Profit is predicted by a Linear Regression model using Budget, Genre, Popularity, Release Month and Runtime. R squared is used to evaluate the accuracy of this model.

By the end of this project, we hope to gain insights on what factors play into a movie’s success and how effective our models are.



# How to run  

To run this project, you will need to download the movies CSV file from the link below and use it as first input argument. No other arguments are required to run this project.
https://www.kaggle.com/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv

I have also added the file to the project repository for ease of use.


