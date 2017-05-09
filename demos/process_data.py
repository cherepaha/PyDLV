import os
from pydlv import data_reader, derivative_calculator

dr = data_reader.DataReader()
dc = derivative_calculator.DerivativeCalculator()

data = dr.read_data(path='../../../data')
data = dr.preprocess_data(data, rewards_sum = [12, 15, 25])

data = dc.append_derivatives(data)

csv_path = 'csv'
if not os.path.exists(csv_path):
    os.makedirs(csv_path)
    
data.to_csv('csv/processed_data_high_low.csv')