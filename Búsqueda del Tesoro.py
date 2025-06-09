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
            next_coord = tuple(np.random.randint(20, size = 2))
            if next_coord not in coords:
                coords.append(next_coord)
    treasure_found = False
    # Repartimos las coordenadas entre los 5 procesos, la coordenada del
    # proceso maestro representará la localización del tesoro y las del
    # resto de nodos la posición inicial de los jugadores.
    key_loc = comm.scatter(coords if rank == 0 else None, root = 0)
    if rank == 0:
        print(f"Tesoro en {key_loc}", flush= True)
    else:
        print(f"Proceso {rank}, casilla inicial {key_loc}", flush= True)

    if rank != 0:
        # Inicializamos una lista con las posiciones que ha visitado cada equipo en los nodos hijo.
        visited_by_team = [key_loc]
        other_teammate = rank + 2 if rank < 3 else rank - 2
        comm.isend(key_loc, dest = other_teammate)
        tm_start_loc = comm.irecv(source = other_teammate).wait()
        visited_by_team.append(tm_start_loc)

    while not treasure_found:
        if rank == 0:
            # Recivimos en el nodo maestro la posición de los distintos jugadores
            # y comprobamos si alguno ha ganado el juego
            for i in range(1,5):
                current_loc = comm.recv(source = i)
                if current_loc == key_loc:
                    winning_team = "Par" if i % 2 == 0 else "Impar"
                    print(f"Gana el equipo {winning_team}", flush= True)
                    treasure_found = comm.bcast(True, root = 0)
                else:
                    comm.send(False, dest = i, tag = 0)
        else:
            # Escogemos una dirección aleatoria hasta que encontramos un movimiento legal.
            # Es decir, un movimiento que no se salga del tablero y que no vaya a una casilla que
            # por la que ya haya pasado alguien de su equipo.
            legal_move = False
            tried_directions = 0
            while not legal_move and tried_directions < 4:
                direction = np.random.randint(4)
                match direction:
                    case 0:
                        test_loc = (key_loc[0] + 1, key_loc[1])
                        tried_directions += 1
                    case 1:
                        test_loc = (key_loc[0] - 1, key_loc[1])
                        tried_directions += 1
                    case 2:
                        test_loc = (key_loc[0], key_loc[1] + 1)
                        tried_directions += 1
                    case _:
                        test_loc = (key_loc[0], key_loc[1] - 1)
                        tried_directions += 1
                if max(test_loc) < 20 and min(test_loc) >= 0 and (test_loc not in visited_by_team or tried_directions >= 4):
                    legal_move = True
                    key_loc = test_loc

            print(f"Proceso {rank}, nueva casilla {key_loc}", flush= True)
            # Enviamos la nueva posición al nodo maestro
            # para comprobar si hemos encontrado el tesoro.
            comm.send(key_loc, dest = 0, tag = 0)
            have_i_found_it = comm.recv(source = 0)
            # Si no se ha encontrado, actualizamos la lista de posiciones comprobadas
            if not have_i_found_it:
                visited_by_team.append(key_loc)
                comm.isend(key_loc, dest = other_teammate)
                tm_visited_loc = comm.irecv(source = other_teammate).wait()
                visited_by_team.append(tm_visited_loc)
