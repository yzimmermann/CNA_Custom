from utils import *

cora = load_dataset("Cora")
visualize_data(cora, "Cora")

data = generate_grid_graph(5, 5)
visualize_data(data, "Grid Graph")







