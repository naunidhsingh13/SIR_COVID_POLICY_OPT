from flask import Flask, render_template
from flask import request
import random
from Processor import Processor

app = Flask(__name__)

enf_values = [i for i in range(0, 105, 5)]
processor = Processor()

@app.route('/')
def main():
    return render_template('index.html', enf_values=enf_values)

@app.route('/handle_data', methods=['POST'])
def handle_data():

    selected_policy = []
    enf = []
    selected_state = None

    """
    Policies: mask, social_distance, transit_stations, groc_pharma,
              retail_recreation, sd_intent, mask_intent, workplace, 
              parks
    """

    if "state" in request.form:
        selected_state = request.form['state']
    else:
        return render_template('index.html', enf_values=enf_values, warn="selecting state is mandatory")

    if "policy" in request.form:
        selected_policy = request.form.getlist('policy')
    if selected_policy:
        for i in selected_policy:
            enf.append(request.form[i])

    print("\nSelected Data:\n")
    print(selected_state, selected_policy, enf)
    log_d1, log_d2, full_d1, full_d2 = processor.get_state_analysis_with_policy_list(selected_state,
                                                                                     selected_policy, enf)
    data1 = log_d1
    data2 = log_d2
    data3 = full_d2


    return render_template('index.html', results="Analysis for", state=selected_state, data1=data1, data2=data2, enf_values=enf_values)


if __name__ == "__main__":
    app.run(debug=True, port=1234, host="0.0.0.0")