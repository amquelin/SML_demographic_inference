# abc_rejection.R

library(readr)
library(abc)
library(parallel)

run_abc_rejection <- function(prefix, Train_val, tol, n_iterations, n_cores, export_path) {

  path_X_train_std <- paste0("./X_train_std.csv")
  path_X_val_std <- paste0("./X_val_std.csv")

  path_Y_train <- paste0("./Y_train.csv")
  path_Y_val <- paste0("./Y_val.csv")
  
  path_X_test <- paste0("./X_test.csv")
  path_X_test_std <- paste0("./X_test_std.csv")
  
  X_train_std <- read.csv(path_X_train_std)
  Y_train <- read.csv(path_Y_train)
  X_test_std <- read.csv(path_X_test_std)
  
  if (Train_val) {
    X_train_std <- rbind(X_train_std, read.csv(path_X_val_std))
    Y_train <- rbind(Y_train, read.csv(path_Y_val))
  }
  
  constant_cols <- sapply(X_train_std, function(col) is.numeric(col) && sd(col) == 0)
  X_train_std <- X_train_std[, !constant_cols]
  X_test_std <- X_test_std[, colnames(X_train_std)]
  
  process_iteration <- function(i) {
  
    cat("iteration", i, "on", n_iterations, "\n")
    
    res_rejection <- abc(
      target = X_test_std[i, ],
      param = Y_train,
      sumstat = X_train_std,
      tol = tol,
      method = "rejection"
    )
    mean_posterior_rejection <- mean(unlist(res_rejection$unadj.values))
    median_posterior_rejection <- median(unlist(res_rejection$unadj.values))
    
    rm(res_rejection)    
    
    return(data.frame(index = i, mean_posterior = mean_posterior_rejection, median_posterior = median_posterior_rejection))
  }
      output_dir <- paste0(export_path, prefix)
    if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
        output_rejection_path <- paste0(output_dir, "/res_rejection_tol_", gsub("\\.", "", as.character(tol)), ".csv")
    write.csv(res_rejection_tol, file = output_rejection_path, row.names = FALSE)
  }

