#############################################################
# This R script downloads a 10M subset of the MovieLens 
# dataset, then creates a training set (edx) and validation
# set. The output is a file submission.csv that contains
# the predicted ratings for the user+movie combinations in
# the file.
#
# The predicted rating, Y, is the sum
# Y = mu + b_i + b_u + eps
# where mu is the overall average, b_i is the movie effect,
# b_u is the user effect, and eps is a remainder term
# which is treated as an error term.
#
# Author: Claudia Valdeavella
# Date: 14 Jan 2019
#############################################################

# Note: this process could take a couple of minutes
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Version 1.10.4-3 of data.table package requires sep to be a single character but the file separator has two characters
# To go around this issue, we will define empty columns that will be removed in the next step
ratings <- fread(unzip(dl, "ml-10M100K/ratings.dat"), sep=":", col.names=c("userId", "rem1", "movieId", "rem2", "rating", "rem3", "timestamp"))
ratings <- select(ratings, -c(rem1,rem2,rem3))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Learners will develop their algorithms on the edx set
# For grading, learners will run algorithm on validation set to generate ratings

validation_rating <- validation$rating
validation <- validation %>% select(-rating)

# Ratings will go into the CSV submission file below:

write.csv(validation %>% select(userId, movieId) %>% mutate(rating = NA),
          "submission.csv", na = "", row.names=FALSE)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Define a function which calculates the residual mean squared error (RMSE)
# metric of the quality of the model
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Our model is Y(u,i) = mu + b_i + b_u + eps
# where mu is the overall average rating, b_i and b_u
# are the movie and user effects on the rating, 
# and eps is a random error
mu <- mean(edx$rating)

movie_avgs <- edx %>%
  group_by(movieId) %>% 
  summarise(b_i = mean(rating - mu))

user_avgs <- edx %>%
  left_join(movie_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

# Refine the model with a tuning parameter lambda that effectively
# penalizes large estimates that come from small sample sizes.
# Estimate lambda using cross-validation and the rmse as the metric
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l) {
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n() + l))
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n() + l))
  predicted_ratings <- edx %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(edx$rating, predicted_ratings))
})

lambda <- lambdas[which.min(rmses)]

# Recalculate b_i and b_u using the best estimate
# for lambda and update the values of b_i in the movie_avgs dataframe
# and b_u in the user_avgs dataframe

movie_avgs <- edx %>%
  group_by(movieId) %>% 
  summarise(b_i = sum(rating - mu)/(n() + lambda))

user_avgs <- edx %>%
  left_join(movie_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - b_i - mu)/(n() + lambda))

# Calculate the predicted ratings for the movie and user
# combinations in the validation set
val_set <- read.csv("submission.csv")
predicted_rating <- val_set %>% 
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  mutate(rating = mu + b_i + b_u) %>%
  .$rating

# Round off the predicted ratings to the nearest half
# or full integers
val_set$rating <- round_any(predicted_rating, 0.5, f = round)

# Apply the lower and upper bounds on the ratings
# Valid ratings are [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
val_set$rating <- ifelse(val_set$rating < 0.5, 0.5, val_set$rating)
val_set$rating <- ifelse(val_set$rating > 5, 5, val_set$rating)

# Update the submission.csv file with the predicted ratings
write.csv(val_set, file="submission.csv", row.names=FALSE)

# Output the rmse for the validation set
rmse <- RMSE(validation_rating, val_set$rating)
pretty_rmse <- round_any(rmse, 0.0001, round)
cat("rmse: ", pretty_rmse, sep=" ")