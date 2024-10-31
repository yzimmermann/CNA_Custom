install:
		conda env create -f environment.yml

demo_fast_kmeans:
		python -m comparing/fast_kmeans.py

demo_relu_rational:
		python -m comparing/relu_rational_cora.py

test_coverage:
		coverage run -m xxxxxxx
