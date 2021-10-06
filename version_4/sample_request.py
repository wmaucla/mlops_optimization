import requests
import timeit

print(timeit.timeit(lambda: requests.get("http://0.0.0.0:8000/"), number=10))
