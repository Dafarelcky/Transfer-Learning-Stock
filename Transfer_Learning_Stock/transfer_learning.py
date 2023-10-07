import datetime
import torch as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# !pip install neuralforecast
from neuralforecast.data.datasets.m4 import M4
from neuralforecast.data.tsdataset import WindowsDataset
from neuralforecast.data.tsloader import TimeSeriesLoader
from neuralforecast.models.mqnhits.mqnhits import MQNHITS

from statsforecast.utils import AirPassengers

t.cuda.is_available() 
#Should be true

# Change this to your own data
Y_df, _, _ = M4.load(directory='./', group='Hourly')

# Normalization
Y_df['y'] = Y_df[['unique_id','y']].groupby('unique_id').transform(lambda x: (x - x.mean()) / x.std())

Y_df.head()

u_id = 'H11'
x_plot = pd.to_datetime(Y_df[Y_df.unique_id==u_id].ds).values
y_plot = Y_df[Y_df.unique_id==u_id].y.values

fig = plt.figure(figsize=(10, 5))
fig.tight_layout()
plt.plot(y_plot)
plt.xlabel('Date', fontsize=17)
plt.ylabel(u_id, fontsize=17)

plt.grid()
plt.show()
plt.close()

forecast_horizon = 24

model = MQNHITS(n_time_in=3*forecast_horizon,
                n_time_out=forecast_horizon,
                quantiles=[5, 50, 95],
                shared_weights=False,
                initialization='lecun_normal',
                activation='ReLU',
                stack_types=3*['identity'],
                n_blocks=3*[1],
                n_layers=3*[2],
                n_mlp_units=3*[2*[256]],
                n_pool_kernel_size=3*[1],
                n_freq_downsample=[12,4,1],
                pooling_mode='max',
                interpolation_mode='linear',
                batch_normalization=False,
                dropout_prob_theta=0,
                learning_rate=0.001,
                lr_decay=1.0,
                lr_decay_step_size=100_000,
                weight_decay=0.0,
                loss_train='MQ',
                loss_valid='MQ',
                frequency='H',
                n_x=0,
                n_s=0,
                n_x_hidden=0,
                n_s_hidden=0,
                loss_hypar=0.5,
                random_seed=1)

train_dataset = WindowsDataset(Y_df=Y_df, X_df=None, S_df=None, 
                               mask_df=None, f_cols=[],
                               input_size=3*forecast_horizon,
                               output_size=forecast_horizon,
                               sample_freq=1,
                               complete_windows=True,
                               verbose=False)

train_loader = TimeSeriesLoader(dataset=train_dataset,
                                batch_size=32,
                                n_windows=1024,
                                shuffle=True)

gpus = -1 if t.cuda.is_available() else 0
trainer = pl.Trainer(max_epochs=None, 
                     max_steps=1000,
                     gradient_clip_val=1.0,
                     progress_bar_refresh_rate=1,
                     gpus=gpus,
                     log_every_n_steps=1)

trainer.fit(model, train_loader)
     
Y_df_plot = Y_df[Y_df['unique_id']=='H1']

plot_dataset = WindowsDataset(Y_df=Y_df_plot, X_df=None, S_df=None, 
                               mask_df=None, f_cols=[],
                               input_size=3*forecast_horizon,
                               output_size=forecast_horizon,
                               sample_freq=1,
                               complete_windows=True,
                               verbose=False)

plot_loader = TimeSeriesLoader(dataset=plot_dataset,
                                batch_size=1,
                                shuffle=False)

outputs = trainer.predict(model, plot_loader)
y_true, y_hat, mask = [t.cat(output).cpu().numpy() for output in zip(*outputs)]

window = 200
plt.plot(y_true[window], c='black', label='True')
plt.plot(y_hat[window,:,0], c='blue')
plt.plot(y_hat[window,:,1], c='blue')
plt.plot(y_hat[window,:,2], c='blue')
plt.fill_between(x=range(forecast_horizon),
                 y1=y_hat[window,:,0],
                 y2=y_hat[window,:,2],
                 alpha=0.2, label='p5-p95')

plt.xlabel('Time', fontsize=17)
plt.ylabel('Prediction', fontsize=17)
plt.grid()
plt.legend()

model_name = 'nhits_3b_m4_hourly'
trainer.save_checkpoint(f"{model_name}.ckpt")

# DO NOT MODIFY
def compute_ds_future(ds, fh):
    freq = ds[-1] - ds[-2]
    ds_future = [ds[-1] + (i + 1) * freq for i in range(fh)]
    return ds_future

# DO NOT MODIFY
def forecast_pretrained_model(ckpt_dir, timestamps, values, fh, max_steps):

    values_mean = np.mean(values)
    values_std = np.std(values)
    values_normalized = (values-values_mean)/values_std
    
    # Parse data
    Y_df = pd.DataFrame({'unique_id': model, 
                         'ds': timestamps, 
                         'y': values_normalized})

    # Model
    mqnhits = MQNHITS.load_from_checkpoint(ckpt_dir)

    # Fit
    if max_steps > 0:
        train_dataset = WindowsDataset(Y_df=Y_df, X_df=None, S_df=None, 
                                       mask_df=None, f_cols=[],
                                       input_size=mqnhits.n_time_in,
                                       output_size=mqnhits.n_time_out,
                                       sample_freq=1,
                                       complete_windows=True,
                                       verbose=False)

        train_loader = TimeSeriesLoader(dataset=train_dataset,
                                        batch_size=1,
                                        n_windows=32,
                                        shuffle=True)
        trainer = pl.Trainer(max_epochs=None, 
                             max_steps=max_steps,
                             gradient_clip_val=1.0,
                             progress_bar_refresh_rate=1, 
                             log_every_n_steps=1)

        trainer.fit(mqnhits, train_loader)
    
    # Forecast
    forecast_df = mqnhits.forecast(Y_df=Y_df)
    forecast_df['y_5'] = values_std*forecast_df['y_5'] + values_mean
    forecast_df['y_50'] = values_std*forecast_df['y_50'] + values_mean
    forecast_df['y_95'] = values_std*forecast_df['y_95'] + values_mean

    # Forecast
    forecast = forecast_df['y_50'].values
    lo = forecast_df['y_5'].values
    hi = forecast_df['y_95'].values
    if fh <= len(forecast):
        forecast = forecast[:fh]
        lo = lo[:fh]
        hi = hi[:fh]
    else:
        forecast = np.tile(forecast, fh)[:fh]
        lo = np.tile(lo, fh)[:fh]
        hi = np.tile(hi, fh)[:fh]

    results = {
        'timestamp': compute_ds_future(timestamps, fh),
        'value': forecast.tolist(),
        'lo': lo.tolist(),
        'hi': hi.tolist()
    }

    return results

# DO NOT MODIFY
timestamps = [datetime.date(month=1,year=i+2022,day=1) for i in range(len(AirPassengers)-forecast_horizon)]
results = forecast_pretrained_model(ckpt_dir=f'./{model_name}.ckpt',
                                    timestamps=timestamps,
                                    values=list(AirPassengers[:-forecast_horizon]),
                                    fh=forecast_horizon,
                                    max_steps=10)

plt.plot(range(len(AirPassengers)), AirPassengers, c='black', label='True')
plt.plot(range(len(AirPassengers)-forecast_horizon,len(AirPassengers)), results['lo'], c='blue')
plt.plot(range(len(AirPassengers)-forecast_horizon,len(AirPassengers)), results['value'], c='blue')
plt.plot(range(len(AirPassengers)-forecast_horizon,len(AirPassengers)), results['hi'], c='blue')
plt.fill_between(x=range(len(AirPassengers)-forecast_horizon,len(AirPassengers)),
                 y1=results['lo'],
                 y2=results['hi'],
                 alpha=0.2, label='p5-p95')

plt.xlabel('Time', fontsize=17)
plt.ylabel('Prediction', fontsize=17)
plt.grid()
plt.legend()                     
                                    