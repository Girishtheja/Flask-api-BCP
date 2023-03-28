from flask import Flask, request, jsonify
import pickle
import numpy as np


model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello"


@app.route('/predict', methods=['POST'])
def predict():
    Time = request.form.get('Time')
    radius_mean = request.form.get('radius_mean')
    texture_mean = request.form.get('texture_mean')
    smoothness_mean = request.form.get('smoothness_mean')
    compactness_mean = request.form.get('compactness_mean')
    concave_points_mean = request.form.get('concave_points_mean')
    symmetry_mean = request.form.get('symmetry_mean')
    fractal_dimension_mean = request.form.get('fractal_dimension_mean')
    texture_se = request.form.get('texture_se')
    perimeter_se = request.form.get('perimeter_se')
    smoothness_se = request.form.get('smoothness_se')
    concavity_se = request.form.get('concavity_se')
    concave_points_se = request.form.get('concave_points_se')
    symmetry_se = request.form.get('symmetry-se')
    fractal_dimension_se = request.form.get('fractional_dimension_se')
    smoothness_worst = request.form.get('smoothness_worst')
    compactness_worst = request.form.get('compactness_worst')
    concave_points_worst = request.form.get('concave_points_worst')
    symmetry_worst = request.form.get('symmetry_worst')
    Tumor_Size = request.form.get('Tumor_Size')
    Lymph_node_status = request.form.get('Lymph_node_status')

    # # result = {'Time': Time,
    # #           'radius_mean': radius_mean,
    # #           'texture_mean': texture_mean,
    # #           'smoothness_mean': smoothness_mean,
    # #           'compactness_mean': compactness_mean,
    # #           'concave_points_mean': concave_points_mean,
    # #           'symmetry_mean': symmetry_mean,
    # #           'fractal_dimension_mean': fractal_dimension_mean,
    # #           'texture_se': texture_se,
    # #           'perimeter_se': perimeter_se,
    # #           'smoothness_se': smoothness_se,
    # #           'concavity_se': concavity_se,
    # #           'concave_points_se': concave_points_se,
    # #           'symmetry_se': symmetry_se,
    # #           'fractal_dimension_se': fractal_dimension_se,
    # #           'smoothness_worst': smoothness_worst,
    # #           'compactness_worst': compactness_worst,
    # #           'concave_points_worst': concave_points_worst,
    # #           'symmetry_worst': symmetry_worst,
    # #           'Tumor_Size': Tumor_Size,
    # #           'Lymph_node_status': Lymph_node_status}
    input_query = np.array([[Time, radius_mean,
                             texture_mean,
                             smoothness_mean,
                             compactness_mean,
                             concave_points_mean,
                             symmetry_mean,
                             fractal_dimension_mean,
                             texture_se,
                             perimeter_se,
                             smoothness_se,
                             concavity_se,
                             concave_points_se,
                             symmetry_se,
                             fractal_dimension_se,
                             smoothness_worst,
                             compactness_worst,
                             concave_points_worst,
                             symmetry_worst,
                             Tumor_Size,
                             Lymph_node_status]])
    result = model.predict(input_query)[0]
    return json.dumps({'Cancer': result})
    # return "Hello"
    # return "hello"


if __name__ == '__main__':
    app.run(debug=True)
