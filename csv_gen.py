import pickle, pandas
from datetime import datetime
from numpy import vectorize

strptime = vectorize(datetime.strptime,excluded='format')

f = pickle.load(open("mvp_fed_speeches","rb"))
f.sort_values(by='date',key=lambda d: strptime(d,'%m/%d/%Y')).to_csv('speeches.csv')