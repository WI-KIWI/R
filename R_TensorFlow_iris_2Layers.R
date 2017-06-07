# load the libraries
library(tensorflow)
library(dummies)

# Create the model
x <- tf$placeholder(tf$float32, shape(NULL, 4L))
h1 <- tf$Variable(tf$random_normal(shape(4L, 3L)))
h2 <- tf$Variable(tf$random_normal(shape(3L, 3L)))
h_out <- tf$Variable(tf$random_normal(shape(3L, 3L)))

b1 <- tf$Variable(tf$random_normal(shape(3L)))
b2 <- tf$Variable(tf$random_normal(shape(3L)))
b_out <- tf$Variable(tf$random_normal(shape(3L)))

layer_1 <- tf$add(tf$matmul(x, h1), b1)
layer_1 = tf$nn$tanh(layer_1)

layer_2 <- tf$add(tf$matmul(layer_1, h2), b2)
layer_2 = tf$nn$tanh(layer_2)

y <- tf$nn$softmax(tf$matmul(layer_2, h_out) + b_out)

# Define loss and optimizer
y_ <- tf$placeholder(tf$float32, shape(NULL, 3L))
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * log(y), reduction_indices=1L))
train_step <- tf$train$GradientDescentOptimizer(0.05)$minimize(cross_entropy)

# Create session and initialize  variables
sess <- tf$Session()
sess$run(tf$global_variables_initializer())

# Load iris data   
datasets <- tf$contrib$learn$datasets
iris <- datasets$load_iris()

# a custom function for min-max-scaler
MinMaxScaler <- function(df, cols) {
   result <- df # make a copy of the input data frame
 
   for (j in cols) { # each specified col
     max_value <- max(df[,j]) # column mean
     min_value <- min(df[,j])
 
     for (i in 1:nrow(result)) { # each row of cur col
       result[i,j] <- (result[i,j] - min_value) / (max_value -    min_value)
     }
   }
   return(result)
 }
cols <- c(1,2,3,4)
iris_feature <- iris$data
iris_feature_scaled <- MinMaxScaler(iris_feature, cols)

# split the data: Training- and test data
set.seed(42) # Set Seed so that same sample can be reproduced in future also 
sample <- sample.int(n = nrow(iris$data), size = floor(.8*nrow(iris$data)), replace = F)
#X_train <- iris$data[sample, ]
#X_test  <- iris$data[-sample, ]
X_train <- iris_feature_scaled[sample, ]
X_test  <- iris_feature_scaled[-sample, ]
y_train_ar <- iris$target[sample]
y_test_ar  <- iris$target[-sample]

target_dummy <- dummy(iris$target, sep = ".")
y_train <- target_dummy[sample, ]
y_test  <- target_dummy[-sample, ]

# Train
for (i in 1:3000) {
  for (j in 1:10) { 
      batch_xs <- X_train[((j-1)*12+1):(j*12),]
      batch_ys <- y_train[((j-1)*12+1):(j*12),]
      sess$run(train_step,
           feed_dict = dict(x = batch_xs, y_ = batch_ys))
} }

# Test trained model
correct_prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
print("The prediction accuracy of 80% training data is")
sess$run(accuracy,
         feed_dict = dict(x = X_train, y_ = y_train))
print("The prediction accuracy of 20% test data is")
sess$run(accuracy,
         feed_dict = dict(x = X_test, y_ = y_test))
y_pred = sess$run(tf$argmax(y, 1L), feed_dict=dict(x = X_test))

# compute the confusion matrix
confusionMatrix(y_pred, y_test_ar)