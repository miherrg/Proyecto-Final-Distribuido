import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__":
    if rank == 0:
        # Generamos 5 coordenadas aleatorias distintas que representarán
        # las posiciones de los jugadores y la localización del tesoro.
        coords = []
        while len(coords) < 5:
            next_coord = np.random.randint(20, size = 2)
            if next_coord not in coords:
                coords.append(next_coord)
    # Repartimos las coordenadas entre los 5 procesos, la coordenada del
    # proceso maestro representará la localización del tesoro y las del
    # resto de nodos la posición inicial de los jugadores.
    key_loc = comm.scatter(coords if rank == 0 else None, root = 0)
    print(f"Proceso {rank}, casilla inicial {key_loc}")

