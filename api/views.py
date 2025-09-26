import pickle
import numpy as np
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Wczytanie wag i bias
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

weights = model_data["weights"]
bias = model_data["bias"]

@csrf_exempt
def predict(request):
    if request.method == "POST":
        data = json.loads(request.body)

        # tu dodac WSZYSTKIE potrzebne pola!!!
        levy = float(data.get("levy", 0))
        mileage = float(data.get("mileage", 0))
        prod_year = float(data.get("prod.year", 0))

        X = np.array([[levy, mileage, prod_year]])
        y_pred = np.dot(X, weights) + bias

        return JsonResponse({"predicted_price": float(y_pred)})

    return JsonResponse({"error": "Only POST allowed"}, status=405)
