import pandas as pd
import numpy as np
import os

class DatasetBuilder:
    # Окно признаков = 2000 тиков
    # Окно таргета = 30000 тиков (условно 15 минут)
    def __init__(self, window_size: int = 2000, future_window: int = 30000, vol_coeff: float = 1.5, min_profit_pct: float = 0.0015):
        self.window_size = window_size
        self.future_window = future_window
        self.vol_coeff = vol_coeff
        self.min_profit_pct = min_profit_pct # 0.15%

    def process_file(self, filepath: str) -> pd.DataFrame:
        print(f"Обработка файла {filepath}...")
        
        columns = ['trade_id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match']
        df = pd.read_csv(filepath, names=columns)
        
        print(f"Загружено {len(df)} тиков. Начинаем расчет фичей...")
        
        df['side'] = np.where(df['is_buyer_maker'], -1, 1)
        
        # 1. Фичи скользящего окна
        df['signed_qty'] = df['qty'] * df['side']
        df['quote_vol'] = df['price'] * df['qty']
        
        print("Считаем rolling метрики (это может занять около минуты)...")
        df['volatility'] = df['price'].rolling(self.window_size).std()
        
        rolling_signed_qty = df['signed_qty'].rolling(self.window_size).sum()
        rolling_qty = df['qty'].rolling(self.window_size).sum()
        df['ofi'] = rolling_signed_qty / rolling_qty
        
        rolling_quote_vol = df['quote_vol'].rolling(self.window_size).sum()
        df['vwap'] = rolling_quote_vol / rolling_qty
        
        df['micro_trend'] = df['price'] - df['price'].shift(self.window_size)
        df['vwap_dev'] = (df['price'] - df['vwap']) / df['price']
        
        # 2. Разметка (Labeling)
        print(f"Формируем таргеты (Горизонт: {self.future_window} тиков, Мин. профит: >{self.min_profit_pct*100}%)...")
        df['future_price'] = df['price'].shift(-self.future_window)
        df['price_diff'] = df['future_price'] - df['price']
        
        # Динамический порог: максимум из (волатильность * коэфф) и (текущая цена * 0.15%)
        # То есть таргет сработает только если профит покроет комиссию + даст прибыль
        df['dynamic_threshold'] = df['volatility'] * self.vol_coeff
        df['min_abs_profit'] = df['price'] * self.min_profit_pct
        df['threshold'] = df[['dynamic_threshold', 'min_abs_profit']].max(axis=1)
        
        df['target'] = 0
        df.loc[df['price_diff'] > df['threshold'], 'target'] = 1
        df.loc[df['price_diff'] < -df['threshold'], 'target'] = -1
        
        df = df.dropna()
        
        # Сохраняем 'price', чтобы бэктестер мог точно посчитать PnL в процентах!
        features = ['ofi', 'volatility', 'vwap_dev', 'micro_trend', 'price', 'target']
        
        print(f"Готово! Сформировано {len(df)} строк с фичами.")
        return df[features]

if __name__ == "__main__":
    builder = DatasetBuilder()
    
    cache_dir = ".data_cache"
    all_dfs = []
    
    for filename in os.listdir(cache_dir):
        if filename.startswith("BTCUSDT-trades-") and filename.endswith(".csv"):
            filepath = os.path.join(cache_dir, filename)
            df_part = builder.process_file(filepath)
            all_dfs.append(df_part)
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Downsampling для PoC (каждая 10-я строка)
        final_df = final_df.iloc[::10]
        output_path = os.path.join(cache_dir, "ml_dataset.csv")
        final_df.to_csv(output_path, index=False)
        print(f"Итоговый датасет сохранен в {output_path}. Размер: {len(final_df)} строк.")
    else:
        print("CSV файлы не найдены.")
