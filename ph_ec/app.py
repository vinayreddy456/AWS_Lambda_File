import pickle
import ee
import numpy as np 
import pandas as pd
#from ipygee import *
import datetime
from statistics import mean
import sys
from datetime import date
import datetime as dt
import os
import csv
from datetime import timedelta 
from datetime import datetime  
import math
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib
from joblib import dump, load
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
    
def handler(event, context):
	#x = float(sys.argv[1])
	#y = float(sys.argv[2])
	x= float("11.474555132531901")
	y = float("76.613916330961004")

	today=dt.date.today()
	ed = today.day
	em = today.month
	ey = today.year
	start = dt.date.today()-dt.timedelta(days=7)
	sd = start.day
	sm= start.month
	sy = start.year
	startTime = datetime(sy, sm, sd)
	endTime = datetime(ey, em, ed)

	#ee.Authenticate()
	#ee.Initialize()

	service_account = 'sagri-846@sagri-map-311113.iam.gserviceaccount.com'
	credentials = ee.ServiceAccountCredentials(service_account, r'creds.json')
	ee.Initialize(credentials)

	# Create image collection
	senti = ee.ImageCollection('COPERNICUS/S2_SR').filterDate(startTime, endTime).filterBounds(ee.Geometry.Point(y, x))
	#senti = senti.map(addNDVI)

	point = ee.Geometry.Point([y, x])
	point1 = point.getInfo()['coordinates']
	p1 = {'type':'Point', 'coordinates':point1}
	info1 = senti.getRegion(p1,500).getInfo()
	len(info1)
	# Reshape image collection 
	header = info1[0]
	data = np.array(info1[1:])
	band_list = ['B1','B2', 'B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']

	iBands = [header.index(b) for b in band_list]
	yData = data[0:,iBands].astype(np.float)
	yData = yData[-1,:].reshape(1,12)

	PredictorScalerFit = load(r'x_scaler.bin')
	ph_y_scaler_fit = load(r'y_PH_scaler.bin')
	ec_y_scaler_fit = load(r'y_EC_scaler.bin')
	# oc_y_scaler_fit = load(r'y_OC_scaler.bin')
	# avn_y_scaler_fit = load(r'y_AVN_scaler.bin')
	# phos_y_scaler_fit = load(r'y_PHOS_scaler.bin')
	# pt_y_scaler_fit = load(r'y_POT_scaler.bin')
	# zn_y_scaler_fit = load(r'y_SULPHUR_scaler.bin')
	# su_y_scaler_fit = load(r'y_ZINC_scaler.bin')
	# fe_y_scaler_fit = load(r'y_IRON_scaler.bin')
	# bo_y_scaler_fit = load(r'y_BORON_scaler.bin')
	# mn_y_scaler_fit = load(r'y_MANGAN_scaler.bin')
	# cu_y_scaler_fit = load(r'y_COPPER_scaler.bin')

	PredictorScalerFit = load(r'x_scaler.bin')
	test_X = PredictorScalerFit.transform(yData)

	with open("ph_xg.dat","rb") as f1:
	  ph_xg_model = pickle.load(f1)

	with open("ec_xg.dat","rb") as f2:
	  ec_xg_model = pickle.load(f2)

	# with open("oc_xg.dat","rb") as f3:
	#   oc_xg_model = pickle.load(f3)

	# with open("avn_xg.dat","rb") as f4:
	#   avn_xg_model = pickle.load(f4)

	# with open("phos_xg.dat","rb") as f5:
	#   phos_xg_model = pickle.load(f5)

	# with open("pt_xg.dat","rb") as f6:
	#   pt_xg_model = pickle.load(f6)

	# with open("su_xg.dat","rb") as f7:
	#   su_xg_model = pickle.load(f7)

	# with open("zn_xg.dat","rb") as f8:
	#   zn_xg_model = pickle.load(f8)

	# with open("bo_xg.dat","rb") as f9:
	#   bo_xg_model = pickle.load(f9)

	# with open("fe_xg.dat","rb") as f10:
	#   fe_xg_model = pickle.load(f10)

	# with open("mn_xg.dat","rb") as f11:
	#   mn_xg_model = pickle.load(f11)

	# with open("cu_xg.dat","rb") as f12:
	#   cu_xg_model = pickle.load(f12)


	ph_Predictions = ph_y_scaler_fit.inverse_transform(ph_xg_model.predict(test_X).reshape(-1,1))
	ec_Predictions = ec_y_scaler_fit.inverse_transform(ec_xg_model.predict(test_X).reshape(-1,1))
	# oc_Predictions = oc_y_scaler_fit.inverse_transform(oc_xg_model.predict(test_X).reshape(-1,1))
	# avn_Predictions = avn_y_scaler_fit.inverse_transform(avn_xg_model.predict(test_X).reshape(-1,1))
	# phos_Predictions = phos_y_scaler_fit.inverse_transform(phos_xg_model.predict(test_X).reshape(-1,1))
	# pt_Predictions = pt_y_scaler_fit.inverse_transform(pt_xg_model.predict(test_X).reshape(-1,1))

	# su_Predictions = su_y_scaler_fit.inverse_transform(su_xg_model.predict(test_X).reshape(-1,1))
	# zn_Predictions = zn_y_scaler_fit.inverse_transform(zn_xg_model.predict(test_X).reshape(-1,1))
	# bo_Predictions = bo_y_scaler_fit.inverse_transform(bo_xg_model.predict(test_X).reshape(-1,1))
	# fe_Predictions = fe_y_scaler_fit.inverse_transform(fe_xg_model.predict(test_X).reshape(-1,1))
	# mn_Predictions = mn_y_scaler_fit.inverse_transform(mn_xg_model.predict(test_X).reshape(-1,1))
	# cu_Predictions = cu_y_scaler_fit.inverse_transform(cu_xg_model.predict(test_X).reshape(-1,1))


	predictions = []
	predictions.append(ph_Predictions[0,0])
	predictions.append(ec_Predictions[0,0])
	# predictions.append(oc_Predictions[0,0])
	# predictions.append(avn_Predictions[0,0])
	# predictions.append(phos_Predictions[0,0])
	# predictions.append(pt_Predictions[0,0])
	# predictions.append(su_Predictions[0,0])
	# predictions.append(zn_Predictions[0,0])
	# predictions.append(bo_Predictions[0,0])
	# predictions.append(fe_Predictions[0,0])
	# predictions.append(mn_Predictions[0,0])
	# predictions.append(cu_Predictions[0,0])

	# .to_csv(r'xg_predictions.csv')x
	return {
	       'statusCode': 200,
	       'body': json.dumps(pd.DataFrame(predictions).to_json(orient="split")),
	       'headers': {
		   "Content-Type": "application/json"
	       }
	}
	    