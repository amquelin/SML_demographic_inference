library(readr)
library(abc)
library(parallel)

run_abc_neuralnet <- function(prefix, Train_val, tol, n_iterations, n_cores, export_path, sizenet) {
  
  path_X_train_std <- "./X_train_std.csv"
  path_X_val_std <- "./X_val_std.csv"
  path_Y_train <- "./Y_train.csv"
  path_Y_val <- "./Y_val.csv"
  path_X_test_std <- "./X_test_std.csv"
  
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

  N <- switch(as.character(tol),
              "0.005" = 300,
              "0.01"  = 300,
              "0.02"  = 300,
              stop("unknown"))  

  correlations <- cor(X_train_std, Y_train, use = "complete.obs")
  if (is.matrix(correlations)) correlations <- rowMeans(abs(correlations))
  sorted_vars <- sort(abs(correlations), decreasing = TRUE)
  top_N_vars <- names(sorted_vars)[1:N]
  
  X_train_std <- X_train_std[, top_N_vars]
  X_test_std <- X_test_std[, top_N_vars]
  
  process_iteration <- function(i) {
    cat("iteration", i, "on", n_iterations, "with tol =", tol, "\n")
      res_neuralnet <- abc(
      target = X_test_std[i, ],
      param = Y_train,
      sumstat = X_train_std,
      tol = tol,
      method = "neuralnet",
      sizenet = 3
    )
    
    mean_posterior_nn <- colMeans(res_neuralnet$unadj.values)
    median_posterior_nn <- apply(res_neuralnet$unadj.values, 2, median)
    
    rm(res_neuralnet)
    
    colnames_mean <- paste0("mean_posterior_", names(mean_posterior_nn))
    colnames_median <- paste0("median_posterior_", names(median_posterior_nn))
    
    return(data.frame(
      index = i,
      setNames(as.list(mean_posterior_nn), colnames_mean),
      setNames(as.list(median_posterior_nn), colnames_median)
    ))
  }
  
  output_dir <- paste0(export_path, prefix)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  output_file_path <- paste0(output_dir, "/res_neuralnet_all_tol_", gsub("\\.", "", as.character(tol)), ".csv")
  write.csv(res_neuralnet_all, file = output_file_path, row.names = FALSE)
}

configurations <- list(
  list(prefix = "demo_para", Train_val = TRUE, tol = 0.0005, n_iterations = n_iterations, n_cores = n_cores, export_path = ""),
  list(prefix = "demo_para", Train_val = TRUE, tol = 0.001, n_iterations = n_iterations, n_cores = n_cores, export_path = "")
)

for (config in configurations) {
  do.call(run_abc_neuralnet, config)
}

