import os
import sys

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from fbprophet import Prophet
from sqlalchemy import create_engine
from matplotlib import pyplot as plt

top_coins_sql = """
    SELECT
        id, market_cap_usd
    FROM 
        (SELECT DISTINCT ON (id) * FROM currency_tick 
             ORDER BY id, last_updated DESC) AS x
             WHERE volume_usd_24h > 10000
    ORDER BY 
        market_cap_usd DESC
    LIMIT 100;
"""

engine = create_engine('postgresql://postgres@localhost:5432/crypto')
coins = pd.read_sql_query(top_coins_sql, engine)

for coin in coins['id']:
    df = pd.read_sql_query(
        "SELECT price_usd, last_updated FROM currency_tick where id = '{}'".format(coin), engine)

    df = df.set_index('last_updated', drop=True)
    df = df.convert_objects(convert_numeric=True)
    df.index = pd.to_datetime(df.index, unit='s')

    df['ds'] = df.index
    df['y'] = df['price_usd']
    df.reset_index()

    m = Prophet()
    m.add_seasonality(name='weekly', period=7, fourier_order=3)
    m.fit(df)

    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    fig = m.plot_components(forecast)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.suptitle('Forecasted {} Price (+{} days) and Historical Seasonality'.format(coin, str(7)))

    if not os.path.exists('./results'):
        os.mkdir('./results')
    if os.path.exists('./results/{}.png'.format(coin)):
        os.remove('./results/{}.png'.format(coin))

    fig.savefig('./results/{}.png'.format(coin))
    plt.close(fig)
