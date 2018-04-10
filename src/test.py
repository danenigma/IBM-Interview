import requests

years_exp = {"yearsOfExperience": 8}
BASE_URL  = "http://127.0.0.1:5000"
response  = requests.post("{}/predict".format(BASE_URL), json = years_exp)
 
print('response: ', response)
