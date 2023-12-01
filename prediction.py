import json
import joblib
import numpy  as np
import pandas as pd

from flask import jsonify

#=== === === === === ===

RESULT           = "result"
RESULT_FRAUD     = "FRAUD"
RESULT_NOT_FRAUD = "NOT-FRAUD"
RESULT_UNKNOWN   = "UNKNOWN"
RESULT_ERROR     = "ERROR"

DATA_EMPTY_VALUE = "-"

def post(request):
    homo_json = ""
    try:
        # getting the JSON data from the request
        data_json = request.json
        
        # getting field features
        names_list, fields_dict = load_fields()
        
        # validating the presence of parameters
        validate(names_list, fields_dict, data_json)
        
        # homologating values
        homo_json = homologate(names_list, fields_dict, data_json)
        
        # getting the prediction 
        result = prediction(homo_json)
        
        # saging the data
        save_data(names_list, data_json, homo_json, result)
        
        response = {RESULT: result}
        return jsonify(response)

    except Exception as ex:
        # throwing the error in result
        error_msg = RESULT_ERROR+': '+str(ex.args[0])
        save_data(names_list, data_json, homo_json, error_msg)

        response  = {RESULT: error_msg}
        return jsonify(response), 400  # 400 Bad Request

#=== === === === === ===
#=== === === === === ===

def get_field_by_name(fields_list, name):
    for field in fields_list:
        if field.get("name") == name:
            return field
    return None

def load_fields():
    fields_path = "data/fields.json"
    with open(fields_path, "r") as json_file:
        fields_list = json.load(json_file)
    
    names_list = [field["name"] for field in fields_list]
    fields_dict = { }
    for name in names_list:
        fields_dict[name] = get_field_by_name(fields_list, name)

    return names_list, fields_dict

#=== === === === === ===

def validate(names_list, fields_dict, data_json):
    params_unknown_list = [ ]
    for key, value in data_json.items():
        if not key in names_list:
            params_unknown_list.append(key)
    if len(params_unknown_list) > 0:
        raise Exception("Unknown Parameters: "+", ".join(params_unknown_list))
    
    params_missed_list = [ ]
    for name in names_list:
        if fields_dict.get(name)["required"] in (True, "true"):
            if not name in data_json:
                params_missed_list.append(name)
            elif is_null_or_empty(data_json[name]):
               params_missed_list.append(name)
    if len(params_missed_list) > 0:
        raise Exception("Required Parameters: "+", ".join(params_missed_list))
    
    params_nums_out_range_list = [ ]
    for name in names_list:
        if name in data_json and not is_null_or_empty(data_json[name]) \
           and fields_dict.get(name)["type"] in ("number", "decimal"):
                if "min" in fields_dict.get(name):
                    if float(data_json[name]) < int(fields_dict.get(name)["min"]):
                        params_nums_out_range_list.append(name)
                if "max" in fields_dict.get(name):
                    if float(data_json[name]) > int(fields_dict.get(name)["max"]):
                        params_nums_out_range_list.append(name)
    if len(params_nums_out_range_list) > 0:
        raise Exception("Out of Range Parameters: "+", ".join(params_nums_out_range_list))
    
    return names_list, fields_dict


def is_null_or_empty(obj):
    if obj is None:
        return True
    elif isinstance(obj, str) and not obj.strip():
        return True
    elif isinstance(obj, (list, tuple, dict, set)) and not obj:
        return True
    else:
        return False

#=== === === === === ===

def homologate(names_list, fields_dict, data_json):
    homo_json = { }
    for name in names_list:
        # getting the final name to send to the model
        homo_name = name
        if 'encoded_name' in fields_dict[name]: 
            homo_name = fields_dict[name]['encoded_name']
        
        # if element is not sended, set null and continue
        if not name in data_json:
            homo_json[homo_name] = np.nan
            continue
        
        # if the value is null or empty, set null and continue
        value = data_json[name]
        if is_null_or_empty(value):
            homo_json[homo_name] = np.nan
            continue                        

        if 'encoded_func' in fields_dict[name] and fields_dict[name]['encoded_func']:
            variables = { 'value': value }
            exec(fields_dict[name]['encoded_func'], globals(), variables)
            result = variables.get('result')
            homo_json[homo_name] = result
            #continue

        # if there are not encoded list, set the value as float and continue
        if not 'encoded_list' in fields_dict[name]:
            homo_json[homo_name] = float(value)
            continue            
        
        # getting the encoded list
        values_dict = { }
        list_filename = "models/"+fields_dict[name]['encoded_list']
        with open(list_filename, "r") as json_file:
            values_dict = json.load(json_file)

        # if value is found, set the value as float and continue
        if value in values_dict:
            homo_json[homo_name] = float(values_dict[value])
            continue
        elif str(value)+".0" in values_dict:
            homo_json[homo_name] = float(values_dict[str(value)+".0"])
            continue
        
        # reached here, nothing more to do, set null
        homo_json[homo_name] = np.nan
        
    #print("data_json = ", data_json)
    #print("homo_json = ", homo_json)

    # returning the homologated JSON
    return homo_json

#=== === === === === ===

def prediction(homo_json):
    model_filename = 'models/model_xgbv3_joblib.pkl'
    model_loaded = joblib.load(model_filename)

    homo_json["TransactionID"] = 0
    dataframe_X = pd.DataFrame(homo_json, index=[0])
    dataframe_X.set_index("TransactionID", inplace=True)
    dataframe_X = dataframe_X[model_loaded.get_booster().feature_names]
    y_pred = model_loaded.predict(dataframe_X)   

    result = str(y_pred[0])
    return RESULT_FRAUD if result == "1" else RESULT_NOT_FRAUD if result == "0" else RESULT_UNKNOWN

#=== === === === === ===

def save_data(names_list, data_json, homo_json, result):
    data_filename = "data/data.csv"
    with open(data_filename, 'a') as file:
        result_normalized = '' if is_null_or_empty(result) else result.replace(',',';')
        homo_strg = str(homo_json).replace(',','; ').replace('\'','').replace('\"','')
        new_line = result_normalized + ', ' + json_csv_line(names_list, data_json) + ', ' + homo_strg
        file.write('\n'+new_line)

def json_csv_line(names_list, data_json):
    values_list = [ ]
    for name in names_list:
        if name in data_json:
            if is_null_or_empty(data_json[name]):
                values_list.append(DATA_EMPTY_VALUE)
            else: values_list.append(str(data_json[name]).replace(',',';'))
        else: values_list.append(DATA_EMPTY_VALUE)
    return ', '.join(values_list)

#=== === === === === ===