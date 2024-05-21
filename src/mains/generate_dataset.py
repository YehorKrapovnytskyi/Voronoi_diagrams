from dataset_generator.VoronoiGenerator import *

vor_gen = VoronoiGenerator("../voronoi_data/")
vor_gen.generate_voronoi_dataset(N_images=1000)