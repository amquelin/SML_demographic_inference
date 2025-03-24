# abc_loclinear.R

library(readr)
library(abc)
library(parallel)

run_abc_loclinear <- function(prefix, Train_val, tol, n_iterations, n_cores, export_path) {
  
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
  
  N <- switch(
    as.character(tol),
    "0.05" = 400,
    "0.02" = 119,
    stop("unknown")
  )
  
  correlations <- cor(X_train_std, Y_train, use = "complete.obs")
  if (is.matrix(correlations)) correlations <- correlations[, 1]
  
  sorted_vars <- sort(abs(correlations), decreasing = TRUE)
  top_N_vars <- names(sorted_vars)[1:N]
  X_train_std_N <- X_train_std[, top_N_vars]
  X_test_std_N <- X_test_std[, top_N_vars]
  
  process_iteration <- function(i) {
    
    cat("iteration", i, "on", n_iterations, "\n")
    
    res_loclinear <- abc(
      target = X_test_std_N[i, ],
      param = Y_train,
      sumstat = X_train_std_N,
      tol = tol,
      method = "loclinear"
    )
    
    mean_posterior_loclinear <- colMeans(res_loclinear$unadj.values)
    median_posterior_loclinear <- apply(res_loclinear$unadj.values, 2, median)
    
    rm(res_loclinear)
    
    colnames_mean <- paste0("mean_posterior_", names(mean_posterior_loclinear))
    colnames_median <- paste0("median_posterior_", names(median_posterior_loclinear))
    
    return(data.frame(
      index = i,
      setNames(as.list(mean_posterior_loclinear), colnames_mean),
      setNames(as.list(median_posterior_loclinear), colnames_median)
    ))
  }
 
  output_dir <- paste0(export_path, prefix)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  output_file_path <- paste0(output_dir, "/res_loclinear_all_tol_", gsub("\\.", "", as.character(tol)), ".csv")
  write.csv(res_loclinear_all, file = output_file_path, row.names = FALSE)
}

configurations <- list(
  list(prefix = "demo_para", Train_val = TRUE, tol = 0.05, n_iterations = n_iterations, n_cores = n_cores, export_path = "")
)

for (config in configurations) {
  do.call(run_abc_loclinear, config)
}

