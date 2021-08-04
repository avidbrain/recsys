import pandas as pd
import numpy as np

ITEM_COL = 'item_id'
USER_COL = 'user_id'
FAKE_ITEM_ID = 999999

# Предфильтрация
def prefilter_items(data, prevalence_range = (0.05, 0.95), price_range = (1.0, 100.0)):
    # Уберем самые популярные товары и самые непопулярные товары
    pop_thr, unpop_thr = prevalence_range
    item_cum_counts = data[ITEM_COL].value_counts().cumsum()
    max_count = item_cum_counts.values[-1]
    top_popular_mask = item_cum_counts < max_count * pop_thr
    top_uppopular_mask = item_cum_counts > max_count * unpop_thr
    blocked_items = item_cum_counts[top_popular_mask | top_uppopular_mask].index
    
    # Уберем товары, которые не продавались за последние 25 недель
    recent_sale_items = data[ITEM_COL][data['week_no'] > data['week_no'].max() - 25]
    old_sale_items = np.setdiff1d(data[ITEM_COL], recent_sale_items)
    blocked_items = np.union1d(blocked_items, old_sale_items)
    
    # Уберем слишком дешевые товары и слишком дорогие товары
    # Цена товара косвенно оценивается по sales_value
    min_price, max_price = price_range
    bad_price_items = (
        data
        .assign(price = lambda x: np.where(x['quantity'] > 0, x['sales_value'] / x['quantity'], 0.0))
        .groupby(ITEM_COL)
        .agg(min_item_price=('price', 'min'), max_item_price=('price', 'max'))
        .query("min_item_price >= @max_price or max_item_price <= @min_price")
        .index
    )
    
    prefiltered_data = data.copy()
    blocked_items = np.union1d(blocked_items, bad_price_items)
    fake_mask = np.isin(data[ITEM_COL], blocked_items)
    prefiltered_data.loc[fake_mask, ITEM_COL] = FAKE_ITEM_ID
    
    return prefiltered_data

def postfilter_items(user_id, recommednations):
    pass