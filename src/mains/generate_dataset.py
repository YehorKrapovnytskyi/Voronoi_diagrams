from dataset_generator.VoronoiGenerator import *

vor_gen = VoronoiGenerator("../voronoi_dataset/full", save_format="coco")
vor_gen.generate_voronoi_dataset(N_images=1000)