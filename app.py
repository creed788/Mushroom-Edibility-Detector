from flask import Flask,render_template,url_for,request
from flask_material import Material
import logging

# EDA PKg
import pandas as pd
import numpy as np

# ML Pkg
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv("data/mushrooms.csv")
    return render_template("preview.html",df_view = df)

@app.route('/',methods=["POST"])
def analyse():
	if request.method == 'POST':
		cap_shape = request.form['cap_shape']
		cap_surface = request.form['cap_surface']
		cap_color = request.form['cap_color']
		bruises = request.form['bruises']
		odor = request.form['odor']
		gill_attachment = request.form['gill_attachment']
		gill_spacing = request.form['gill_spacing']
		gill_size = request.form['gill_size']
		gill_color = request.form['gill_color']
		stalk_shape = request.form['stalk_shape']
		stalk_root = request.form['stalk_root']
		stalk_surface_above_ring = request.form['stalk_surface_above_ring']
		stalk_surface_below_ring = request.form['stalk_surface_below_ring']
		stalk_color_above_ring = request.form['stalk_color_above_ring']
		stalk_color_below_ring = request.form['stalk_color_below_ring']
		veil_color = request.form['veil_color']
		ring_number = request.form['ring_number']
		ring_type = request.form['ring_type']
		spore_print_color = request.form['spore_print_color']
		population = request.form['population']
		habitat = request.form['habitat']
		model_choice = request.form['model_choice']


		# Clean the data by convert from unicode to float
		sample_data = [cap_shape,cap_surface,cap_color,bruises,odor,gill_attachment,
					   gill_spacing,gill_size,gill_color,stalk_shape,stalk_root,
					   stalk_surface_above_ring,stalk_surface_below_ring,
					   stalk_color_above_ring,stalk_color_below_ring,veil_color,ring_number,
					   ring_type,spore_print_color,population,habitat]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# Reloading the Model
		if model_choice == 'dsmodel':
		    ds_model = joblib.load('decision_tree_model.pkl')
		    result_prediction = ds_model.predict(ex1)
		elif model_choice == 'rfmodel':
			nb_model = joblib.load('random_forest_model.pkl')
			result_prediction = nb_model.predict(ex1)
		elif model_choice == 'svmmodel':
			svm_model = joblib.load('suport_vector_machine_model.pkl')
			result_prediction = svm_model.predict(ex1)

	return render_template('index.html',cap_shape='cap_shape',cap_surface='cap_surface',cap_color='cap_color',bruises='bruises',
						   odor='odor',gill_attachment='gill_attachment',gill_spacing='gill_spacing',gill_size='gill_size',gill_color='gill_color',
						   stalk_shape='stalk_shape',stalk_root='stalk_root',stalk_surface_above_ring='stalk_surface_above_ring',
							stalk_color_below_ring='stalk_surface_below_ring',stalk_color_above_ring='stalk_color_above_ring',
						   veil_color='veil_color',ring_number='ring_number',
						   ring_type='ring_type',spore_print_color='spore_print_color',population='population',habitat='habitat',
						    model_selected='model_choice', result_prediction=result_prediction)


if __name__ == '__main__':
	app.run(debug=True)