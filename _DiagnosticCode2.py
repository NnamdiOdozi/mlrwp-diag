"""
_DiagnosticCode.py

This module contains the run_diagnostics() function that:
- Receives the dataset and hyperparameters from main.py.
- Trains the model using a scikit-learn Pipeline.
- Generates diagnostic plots (figures) and computes an evaluation metric (MSE).
- Returns a dictionary containing the MSE and a list of registered figures.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# NOTE: Removed SummaryWriter/TensorBoard imports and usage

# Import custom classes from _DiagnosticClasses
from _DiagnosticClasses import (
    TabularNetRegressor,
    LogLinkForwardNet,
    ColumnKeeper,
)


def run_diagnostics(dat, nn_iter, max_lr, init_bias, n_hidden, batchnorm, dropout, run_name="diagnostic_run"):
    """
    Train the diagnostic model and generate diagnostic plots.

    Parameters:
      dat       : Pandas DataFrame containing the dataset.
      nn_iter   : Number of training epochs.
      max_lr    : Maximum learning rate.
      init_bias : Initial bias value.
      n_hidden  : Number of hidden nodes.
      batchnorm : Boolean indicating if batch normalization is used.
      dropout   : Dropout rate (float).
      run_name  : Name of the run (for MLflow logging).

    Returns:
      A dictionary with:
         "mse"      : Mean Squared Error computed on the training set.
         "figures"  : A list of tuples (figure_name, matplotlib.figure.Figure).
    """
    figures = []

    try:
        # --- Removed TensorBoard Setup ---

        # --- Configuration ---
        list_of_features = [
            "occurrence_time", "notidel", "development_period", "pmt_no",
            "log1_paid_cumulative", "max_paid_dev_factor", "min_paid_dev_factor",
        ]
        output_field = "claim_size"

        # Data Prep
        dat[output_field] = pd.to_numeric(dat[output_field], errors='coerce')
        dat = dat.dropna(subset=[output_field])
        dat = dat[dat[output_field] > 0].copy()
        if dat.empty:
            raise ValueError("Dataset is empty after filtering for positive claim sizes.")

        train_dat = dat[dat['train_ind'] == 1].copy()
        if train_dat.empty:
             raise ValueError("Training dataset is empty after filtering.")

        X_train = train_dat[list_of_features]
        y_train = train_dat[[output_field]]

        # --- Model Pipeline ---
        dropout_rate = dropout

        model_NN = Pipeline(
            steps=[
                ("keep", ColumnKeeper(list_of_features)),
                ('zero_to_one', MinMaxScaler()),
                ("model", TabularNetRegressor(
                    module=LogLinkForwardNet,
                    max_iter=nn_iter,
                    max_lr=max_lr,
                    n_hidden=n_hidden,
                    batch_norm=batchnorm,
                    dropout=dropout_rate,
                    init_bias=init_bias,
                    verbose=1,
                    # Removed writer argument
                ))
            ]
        )

        # --- Train Model ---
        print("Starting model training...")
        model_NN.fit(X_train, y_train)
        print("Model training finished.")

        # --- Evaluate Model (MSE on Training Data) ---
        print("Calculating predictions on training data...")
        y_train_pred = model_NN.predict(X_train)
        mse_train = mean_squared_error(y_train.values.ravel(), y_train_pred.ravel())
        print(f"MSE on Training Data: {mse_train:.4f}")
        # Removed final MSE TensorBoard logging

        # --- Generate Predictions for Plots ---
        print("Calculating predictions on all data for plotting...")
        dat['pred_claims'] = model_NN.predict(dat[list_of_features])
        epsilon = 1e-8
        dat['log_pred_claims'] = np.log(np.maximum(dat['pred_claims'], epsilon))
        dat['log_actual'] = np.log(np.maximum(dat[output_field], epsilon))


        # --- Generate Diagnostic Plots ---

        def make_model_subplots(plot_dat, target_col):
            # (Plotting code remains the same)
            fig, axes = plt.subplots(3, 2, sharex='all', sharey='all', figsize=(15, 15))
            fig.suptitle("Model Performance by Period (Mean Claim Size)", fontsize=16)
            # ... (rest of plotting code) ...
            # Train, Occ
            (plot_dat
                .loc[lambda df: df.train_ind == 1]
                .groupby(["occurrence_time"])
                .agg(actual_mean=(target_col, "mean"), pred_mean=("pred_claims", "mean"))
            ).rename(columns={"actual_mean": target_col}).plot(ax=axes[0,0], logy=True, title="Train, Occurrence")
            axes[0,0].legend([target_col, "Prediction"])
            # Train, Dev
            (plot_dat
                .loc[lambda df: df.train_ind == 1]
                .groupby(["development_period"])
                .agg(actual_mean=(target_col, "mean"), pred_mean=("pred_claims", "mean"))
            ).rename(columns={"actual_mean": target_col}).plot(ax=axes[0,1], logy=True, title="Train, Development")
            axes[0,1].legend([target_col, "Prediction"])
            # Test, Occ
            (plot_dat
                .loc[lambda df: df.train_ind == 0]
                .groupby(["occurrence_time"])
                .agg(actual_mean=(target_col, "mean"), pred_mean=("pred_claims", "mean"))
            ).rename(columns={"actual_mean": target_col}).plot(ax=axes[1,0], logy=True, title="Test, Occurrence")
            axes[1,0].legend([target_col, "Prediction"])
            # Test, Dev
            (plot_dat
                .loc[lambda df: df.train_ind == 0]
                .groupby(["development_period"])
                .agg(actual_mean=(target_col, "mean"), pred_mean=("pred_claims", "mean"))
            ).rename(columns={"actual_mean": target_col}).plot(ax=axes[1,1], logy=True, title="Test, Development")
            axes[1,1].legend([target_col, "Prediction"])
            # All, Occ
            (plot_dat
                .groupby(["occurrence_time"])
                .agg(actual_mean=(target_col, "mean"), pred_mean=("pred_claims", "mean"))
            ).rename(columns={"actual_mean": target_col}).plot(ax=axes[2,0], logy=True, title="All, Occurrence")
            axes[2,0].legend([target_col, "Prediction"])
            # All, Dev
            (plot_dat
                .groupby(["development_period"])
                .agg(actual_mean=(target_col, "mean"), pred_mean=("pred_claims", "mean"))
            ).rename(columns={"actual_mean": target_col}).plot(ax=axes[2,1], logy=True, title="All, Development")
            axes[2,1].legend([target_col, "Prediction"])

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            return fig

        print("Generating Development/Occurrence Subplots...")
        fig_dev_occ = make_model_subplots(dat, output_field)
        figures.append(("Development and Occurrence Performance", fig_dev_occ))
        # Removed writer.add_figure call
        plt.close(fig_dev_occ)

        # Prepare data subsets
        datTest = dat.loc[dat.train_ind == 0].copy()
        datTrain = dat.loc[dat.train_ind == 1].copy()

        # --- Actual vs Expected Plots ---
        print("Generating Actual vs. Expected plots...")

        # AvsE - Train data - All records
        fig_avse_train_all, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(datTrain[output_field], datTrain["pred_claims"], alpha=0.5)
        min_val = min(datTrain[output_field].min(), datTrain["pred_claims"].min())
        max_val = max(datTrain[output_field].max(), datTrain["pred_claims"].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        ax.set_xlabel('Actual Claim Size', fontsize=12); ax.set_ylabel('Predicted Claim Size', fontsize=12)
        ax.set_title('Actual vs. Predicted - Train Data (All Records)'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
        figures.append(("AvsE Train All", fig_avse_train_all))
        # Removed writer.add_figure call
        plt.close(fig_avse_train_all)

        # AvsE - Test data - All records
        fig_avse_test_all, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(datTest[output_field], datTest["pred_claims"], alpha=0.5)
        min_val = min(datTest[output_field].min(), datTest["pred_claims"].min())
        max_val = max(datTest[output_field].max(), datTest["pred_claims"].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        ax.set_xlabel('Actual Claim Size', fontsize=12); ax.set_ylabel('Predicted Claim Size', fontsize=12)
        ax.set_title('Actual vs. Predicted - Test Data (All Records)'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
        figures.append(("AvsE Test All", fig_avse_test_all))
        # Removed writer.add_figure call
        plt.close(fig_avse_test_all)

        # Logged AvsE - Train data - All records
        fig_avse_train_log, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(datTrain["log_actual"], datTrain["log_pred_claims"], alpha=0.5)
        min_val = min(datTrain["log_actual"].min(), datTrain["log_pred_claims"].min())
        max_val = max(datTrain["log_actual"].max(), datTrain["log_pred_claims"].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        ax.set_xlabel('ln(Actual Claim Size + eps)', fontsize=12); ax.set_ylabel('ln(Predicted Claim Size + eps)', fontsize=12)
        ax.set_title('Logged Actual vs. Predicted - Train Data'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
        figures.append(("Logged AvsE Train All", fig_avse_train_log))
        # Removed writer.add_figure call
        plt.close(fig_avse_train_log)

        # Logged AvsE - Test data - All records
        fig_avse_test_log, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(datTest["log_actual"], datTest["log_pred_claims"], alpha=0.5)
        min_val = min(datTest["log_actual"].min(), datTest["log_pred_claims"].min())
        max_val = max(datTest["log_actual"].max(), datTest["log_pred_claims"].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        ax.set_xlabel('ln(Actual Claim Size + eps)', fontsize=12); ax.set_ylabel('ln(Predicted Claim Size + eps)', fontsize=12)
        ax.set_title('Logged Actual vs. Predicted - Test Data'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
        figures.append(("Logged AvsE Test All", fig_avse_test_log))
        # Removed writer.add_figure call
        plt.close(fig_avse_test_log)

        # Ultimates
        dat_ult_train = datTrain.groupby("claim_no", as_index=False).last()
        dat_ult_test = datTest.groupby("claim_no", as_index=False).last()

        # AvsE - Train data - Ultimates only
        fig_avse_train_ult, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(dat_ult_train[output_field], dat_ult_train["pred_claims"], alpha=0.5)
        min_val = min(dat_ult_train[output_field].min(), dat_ult_train["pred_claims"].min())
        max_val = max(dat_ult_train[output_field].max(), dat_ult_train["pred_claims"].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        ax.set_xlabel('Actual Ultimate Claim Size', fontsize=12); ax.set_ylabel('Predicted Ultimate Claim Size', fontsize=12)
        ax.set_title('Actual vs. Predicted - Train Data (Ultimates Only)'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
        figures.append(("AvsE Train Ultimates", fig_avse_train_ult))
        # Removed writer.add_figure call
        plt.close(fig_avse_train_ult)

        # AvsE - Test data - Ultimates only
        fig_avse_test_ult, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(dat_ult_test[output_field], dat_ult_test["pred_claims"], alpha=0.5)
        min_val = min(dat_ult_test[output_field].min(), dat_ult_test["pred_claims"].min())
        max_val = max(dat_ult_test[output_field].max(), dat_ult_test["pred_claims"].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        ax.set_xlabel('Actual Ultimate Claim Size', fontsize=12); ax.set_ylabel('Predicted Ultimate Claim Size', fontsize=12)
        ax.set_title('Actual vs. Predicted - Test Data (Ultimates Only)'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
        figures.append(("AvsE Test Ultimates", fig_avse_test_ult))
        # Removed writer.add_figure call
        plt.close(fig_avse_test_ult)

        # --- QQ Plots (Calibration) ---
        print("Generating QQ calibration plots...")
        n_quantiles = 20
        try:
            dat["pred_claims_quantile"] = pd.qcut(dat["pred_claims"], n_quantiles, labels=False, duplicates='drop')
        except ValueError as e:
            print(f"Warning: Could not create {n_quantiles} quantiles. Trying 10. Error: {e}")
            try:
                 n_quantiles = 10
                 dat["pred_claims_quantile"] = pd.qcut(dat["pred_claims"], n_quantiles, labels=False, duplicates='drop')
            except ValueError:
                 print("Warning: Still unable to create quantiles. Skipping QQ plots.")
                 dat["pred_claims_quantile"] = -1

        if 'pred_claims_quantile' in dat.columns and dat["pred_claims_quantile"].nunique() > 1:
            # Train QQ plot
            qq_train = dat.loc[dat.train_ind == 1].groupby("pred_claims_quantile")[[output_field, "pred_claims"]].mean()
            fig_qq_train, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(qq_train[output_field], qq_train["pred_claims"])
            min_val = min(qq_train[output_field].min(), qq_train["pred_claims"].min())
            max_val = max(qq_train[output_field].max(), qq_train["pred_claims"].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Calibration')
            ax.set_xlabel(f'Mean Actual ({n_quantiles}-Quantile)', fontsize=12); ax.set_ylabel(f'Mean Predicted ({n_quantiles}-Quantile)', fontsize=12)
            ax.set_title(f'Calibration Plot (QQ) - Train Data ({n_quantiles}-Quantiles)'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
            figures.append(("QQ Plot Train", fig_qq_train))
            # Removed writer.add_figure call
            plt.close(fig_qq_train)

            # Test QQ plot
            qq_test = dat.loc[dat.train_ind == 0].groupby("pred_claims_quantile")[[output_field, "pred_claims"]].mean()
            fig_qq_test, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(qq_test[output_field], qq_test["pred_claims"])
            min_val = min(qq_test[output_field].min(), qq_test["pred_claims"].min())
            max_val = max(qq_test[output_field].max(), qq_test["pred_claims"].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Calibration')
            ax.set_xlabel(f'Mean Actual ({n_quantiles}-Quantile)', fontsize=12); ax.set_ylabel(f'Mean Predicted ({n_quantiles}-Quantile)', fontsize=12)
            ax.set_title(f'Calibration Plot (QQ) - Test Data ({n_quantiles}-Quantiles)'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
            figures.append(("QQ Plot Test", fig_qq_test))
            # Removed writer.add_figure call
            plt.close(fig_qq_test)
        else:
            print("Skipping QQ plots due to issues creating quantiles.")

        # --- Commented out Tableau specific code ---
        # ... (remains commented) ...

        print("Diagnostic generation complete.")

    except Exception as e:
         print(f"An error occurred in run_diagnostics: {e}")
         raise e
    # Removed finally block with writer.close()

    # --- Return Results ---
    results = {
        "mse": mse_train if 'mse_train' in locals() else np.nan,
        "figures": figures,
    }
    plt.close('all')
    return results