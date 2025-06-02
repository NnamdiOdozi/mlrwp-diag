
# Contents  
# 
# 1. The general python environment  
# 2. Read in the data 
# 3. Set up the pipeline  
# 4. Define and run the neural network
# 5. Produce some basic outputs 

# 1. The general Python environment


import subprocess

subprocess.run(["pip", "show", "torch", "pandas", "numpy", "scikit-learn"], check=True)


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.autograd import Variable

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, PredefinedSplit
import math

# Set the number of epochs to run

nn_iter = 1000     

pd.options.display.float_format = '{:,.2f}'.format

# 2. Read in the data  
# This dataset has the train and test data splits pre-defined by the train_ind flag.

dirname_in="https://raw.githubusercontent.com/MLRWP/mlrwp-book/main/Research/"
filename_in="datwTestTrainSplit.csv"

dat = pd.read_csv(
    dirname_in + filename_in
)

# 3. Set up the pipeline - define the classes to be used  
# * Define the TabularNetRegressor class
# * Define the ColumnKeeper class
# 
# Define the TabularNetRegressor class

class TabularNetRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self, 
        module,
        criterion=nn.MSELoss(),
        max_iter=nn_iter,   
        max_lr=0.01,
        keep_best_model=False,
        batch_function=None,
        rebatch_every_iter=1,
        n_hidden=20,                  
        l1_penalty=0.0,
        l1_applies_params=["linear.weight", "hidden.weight"],
        weight_decay=0.0,
        batch_norm=False,
        dropout=0.0,
        clip_value=None,
        verbose=1,
        device="cpu",
        init_bias=None,
        **kwargs
    ):
        self.module = module
        self.criterion = criterion
        self.keep_best_model = keep_best_model
        self.l1_penalty = l1_penalty
        self.l1_applies_params = l1_applies_params
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.n_hidden = n_hidden
        self.batch_norm = batch_norm
        self.batch_function = batch_function
        self.rebatch_every_iter = rebatch_every_iter
        self.dropout = dropout
        self.device = device
        self.target_device = torch.device(device)    
        self.max_lr = max_lr
        self.init_bias = init_bias
        self.print_loss_every_iter = max(1, int(max_iter / 10))
        self.verbose = verbose
        self.clip_value = clip_value
        self.kwargs = kwargs

    def fix_array(self, y):
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y = y.astype(np.float32)
        return y

    def setup_module(self, n_input, n_output):
        self.module_ = self.module(
            n_input=n_input, 
            n_output=n_output,
            n_hidden=self.n_hidden,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
            init_bias=self.init_bias_calc if self.init_bias is None else self.init_bias,
            **self.kwargs
        ).to(self.target_device)

    def fit(self, X, y):
        n_input = X.shape[-1]
        n_output = 1 if y.ndim == 1 else y.shape[-1]
        self.init_bias_calc = np.log(y.mean()).values.astype(np.float32)
        self.setup_module(n_input=n_input, n_output=n_output)
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        X_tensor = torch.from_numpy(self.fix_array(X)).to(self.target_device)
        y_tensor = torch.from_numpy(self.fix_array(y)).to(self.target_device)

        optimizer = torch.optim.AdamW(
            params=self.module_.parameters(),
            lr=self.max_lr / 10,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.max_lr, 
            steps_per_epoch=1, 
            epochs=self.max_iter
        )

        try:
            loss_fn = self.criterion(log_input=False).to(self.target_device)
        except TypeError:
            loss_fn = self.criterion

        best_loss = float('inf')

        if self.batch_function is not None:
            X_tensor_batch, y_tensor_batch = self.batch_function(X_tensor, y_tensor)
        else:
            X_tensor_batch, y_tensor_batch = X_tensor, y_tensor

        for epoch in range(self.max_iter):
            self.module_.train()
            y_pred = self.module_(X_tensor_batch)
            loss = loss_fn(y_pred, y_tensor_batch)
            if self.l1_penalty > 0.0:
                loss += self.l1_penalty * sum(
                    [
                        w.abs().sum()
                        for p, w in self.module_.named_parameters()
                        if p in self.l1_applies_params
                    ]
                )

            if self.keep_best_model & (loss.item() < best_loss):
                best_loss = loss.item()
                self.best_model = self.module_.state_dict()

            optimizer.zero_grad()
            loss.backward()

            if self.clip_value is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.module_.parameters(), self.clip_value)
                if (self.verbose >= 1) & (grad_norm > self.clip_value):
                    print(f'Gradient norms have been clipped in epoch {epoch}, value before clipping: {grad_norm}')    

            optimizer.step()
            scheduler.step()

            if torch.isnan(loss.data).tolist():
                raise ValueError('Error: nan loss')

            if (epoch % self.print_loss_every_iter == 0) and (self.verbose > 0):
                self.module_.eval()
                self.module_.point_estimates = True
                y_pred_point = self.module_(X_tensor)
                assert(y_pred_point.size() == y_tensor.size())
                rmse = torch.sqrt(torch.mean(torch.square(y_pred_point - y_tensor)))
                self.module_.train()
                self.module_.point_estimates = False
                print("Train RMSE: ", rmse.data.tolist(), " Train Loss: ", loss.data.tolist(), " Epoch: ", epoch)

            if (self.batch_function is not None) & (epoch % self.rebatch_every_iter == 0):
                print(f"refreshing batch on epoch {epoch}")
                X_tensor_batch, y_tensor_batch = self.batch_function(X_tensor, y_tensor)
        
        if self.keep_best_model:
            self.module_.load_state_dict(self.best_model)
            self.module_.eval()

        return self

    def predict(self, X, point_estimates=True):
        check_is_fitted(self)
        X = check_array(X)
        X_tensor = torch.from_numpy(self.fix_array(X)).to(self.target_device)
        self.module_.eval()
        self.module_.point_estimates = point_estimates
        if point_estimates:
            y_pred = self.module_(X_tensor).cpu().detach().numpy()
            if y_pred.shape[-1] == 1: 
                return y_pred.ravel()
            else:
                return y_pred
        else:
            y_pred = self.module_(X_tensor)
            return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        y = self.fix_array(y)
        return -np.sqrt(np.mean((y_pred - y)**2))

# Define the ColumnKeeper class

class ColumnKeeper(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.copy()[self.cols]

# 4. Define and run the neural network  
# Define the LogLinkForward class

class LogLinkForwardNet(nn.Module):
    def __init__(
        self, 
        n_hidden,
        batch_norm,
        dropout,
        n_input=8,
        n_output=1,
        init_bias=0,
    ): 
        super(LogLinkForwardNet, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.batch_norm = batch_norm
        if batch_norm:
            self.batchn = torch.nn.BatchNorm1d(n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear = torch.nn.Linear(n_hidden, n_output)
        nn.init.zeros_(self.linear.weight)
        self.linear.bias.data = torch.tensor(init_bias)

    def forward(self, x):
        h = F.relu(self.hidden(x))
        if self.batch_norm:
            h = self.batchn(h)
        return torch.exp(self.linear(h))

# Define which variables to include in the model

list_of_features = [
    "claim_no", "occurrence_time", "notidel", "development_period", "pmt_no",
    "log1_paid_cumulative", "max_paid_dev_factor", "min_paid_dev_factor",
]
output_field = ["claim_size"]
youtput = "claim_size"

dat.loc[:, list_of_features + [youtput]]

# Run the model using pipeline

model_NN = Pipeline(
    steps=[
        ("keep", ColumnKeeper(list_of_features)),
        ("zero_to_one", MinMaxScaler()),
        ("model", TabularNetRegressor(LogLinkForwardNet))
    ]
)

model_NN.fit(
    dat.loc[dat.train_ind == 1],
    dat.loc[dat.train_ind == 1, ["claim_size"]]
)

# 5. Produce some basic outputs  

def make_model_subplots(model, dat):
    fig, axes = plt.subplots(3, 2, sharex='all', sharey='all', figsize=(15, 15))

    (dat
        .assign(payment_size_pred = model.predict(dat))
        .loc[lambda df: df.train_ind]
        .groupby(["occurrence_period"])
        .agg({youtput: "mean", "payment_size_pred": "mean"})
    ).plot(ax=axes[0,0], logy=True)
    axes[0,0].title.set_text("Train, Occur")

    (dat
        .assign(payment_size_pred = model.predict(dat))
        .loc[lambda df: df.train_ind]
        .groupby(["development_period"])
        .agg({youtput: "mean", "payment_size_pred": "mean"})
    ).plot(ax=axes[0,1], logy=True)
    axes[0,1].title.set_text("Train, Dev")

    (dat
        .assign(payment_size_pred = model.predict(dat))
        .loc[lambda df: ~df.train_ind]
        .groupby(["occurrence_period"])
        .agg({youtput: "mean", "payment_size_pred": "mean"})
    ).plot(ax=axes[1,0], logy=True)
    axes[1,0].title.set_text("Test, Occ")

    (dat
        .assign(payment_size_pred = model.predict(dat))
        .loc[lambda df: ~df.train_ind]
        .groupby(["development_period"])
        .agg({youtput: "mean", "payment_size_pred": "mean"})
    ).plot(ax=axes[1,1], logy=True)
    axes[1,1].title.set_text("Test, Dev")

    (dat
        .assign(payment_size_pred = model.predict(dat))
        .groupby(["occurrence_period"])
        .agg({youtput: "mean", "payment_size_pred": "mean"})
    ).plot(ax=axes[2,0], logy=True)
    axes[2,0].title.set_text("All, Occ")

    (dat
        .assign(payment_size_pred = model.predict(dat))
        .groupby(["development_period"])
        .agg({youtput: "mean", "payment_size_pred": "mean"})
    ).plot(ax=axes[2,1], logy=True)
    axes[2,1].title.set_text("All, Dev")

make_model_subplots(model_NN, dat)

dat["pred_claims"] = model_NN.predict(dat)

# A vs E scatterplot

plt.scatter(dat[youtput], dat["pred_claims"])
plt.xlabel('Actual')
plt.ylabel('Expected')
plt.plot([0, 3000000], [0, 3000000])

# QQ plots, for training data and test data

dat["pred_claims_decile"] = pd.qcut(dat["pred_claims"], 10, labels=False, duplicates='drop')
dat["pred_claims_20cile"] = pd.qcut(dat["pred_claims"], 20, labels=False, duplicates='drop')

# Train dataset
X_sum = dat.loc[dat.train_ind == 1].groupby("pred_claims_20cile").agg("mean").reset_index()
X_sum = dat.groupby("pred_claims_20cile").agg("mean").reset_index()
plt.scatter(X_sum.claim_size, X_sum.pred_claims)
plt.xlabel('Actual')
plt.ylabel('Expected')
plt.plot([0, 900000], [0, 900000])

# Test dataset
X_sum = dat.loc[dat.train_ind == 0].groupby("pred_claims_20cile").agg("mean").reset_index()
X_sum = dat.groupby("pred_claims_20cile").agg("mean").reset_index()
plt.scatter(X_sum.claim_size, X_sum.pred_claims)
plt.xlabel('Actual')
plt.ylabel('Expected')
plt.plot([0, 900000], [0, 900000])
