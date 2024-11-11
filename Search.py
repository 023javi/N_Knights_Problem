import numpy as np
import pandas as pd

def initial_state(M, N):
    # Crea un tablero vacío usando 0s
  return np.zeros((M, N), dtype=np.int8)

board = initial_state(3, 3)
print("Tablero inicial:")
print(board)

def in_board(board, fila, columna):
    return 0 <= fila < board.shape[0] and 0 <= columna < board.shape[1]

#movimientos posibles de un caballo en el tablero
possible_movements = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

def valid_movements(i, j, board):
    if board[i, j]:  # Si ya hay un caballo, no es un movimiento válido
        return False

    # Verificar que la posición no está amenazada por otro caballo
    for dx, dy in possible_movements:
        x, y = i + dx, j + dy
        if in_board(board, x, y) and board[x, y] == 1:
            return False
    return True

def mark_as_attack(board, i, j):
    for dx, dy in possible_movements:
        x, y = i + dx, j + dy
        if in_board(board, x, y) and board[x, y] == 0:
            board[x, y] = -1  # Marcamos la posición atacada
    return board

def put_knight(board, i, j):
    board[i, j] = 1  # 1 indica que hay un caballo
    mark_as_attack(board, i, j)
    return board

def expand(board):
    boards = []
    M, N = board.shape

    for i in range(M):
        for j in range(N):
            if valid_movements(i, j, board) and board[i, j] == 0:  # Solo expande en posiciones válidas
                new_board = board.copy()
                put_knight(new_board, i, j)
                boards.append(new_board)
    return boards

def is_solution(board):
    # Obtiene las dimensiones del tablero: M para filas y N para columnas
    M, N = board.shape
    sol = None
    # Recorre cada posición en el tablero
    for i in range(M):
        for j in range(N):
            # Verifica si la casilla está libre (representada por 0)
            # y si existen movimientos válidos desde esta posición
            if board[i, j] == 0 and valid_movements(i, j, board) :
                sol = False
                return sol    # Si hay una casilla libre con movimientos válidos, el tablero no es una solución

    # Si se recorren todas las casillas sin encontrar ninguna libre con movimientos válidos,
    # se considera que el tablero es una solución
    sol = True
    return sol

def cost(path):
    cost = 0
    # Verifica si el camino tiene más de un tablero en la secuencia
    if len(path) > 1:
        # Obtiene el tablero penúltimo (estado anterior) y el tablero actual (último estado en path)
          tablero_anterior = path[-2]
          board = path[-1]
          # Calcula el costo como la diferencia en el número de casillas libres (representadas por 0)
          # entre el tablero anterior y el tablero actual
          cost = np.count_nonzero(board == -1) - np.count_nonzero(tablero_anterior == -1)

    # Si solo hay un tablero en el camino, inicializa el costo con el total de casillas
    elif len(path) == 1:
        board = path[-1]
        M, N = board.shape
        cost = (M * N)

    # Si no hay tableros en el camino, el costo es 0
    else:
        cost = 0

    return cost # Retorna el costo calculado


def heuristic_1(board):
    # Calcula la heurística de un tablero aquí
    if len(board)!=0:
        heuristic = np.count_nonzero(board == 0) + np.count_nonzero(board == -1)
        return heuristic
    return 0


def heuristic_2(board):
    M, N = board.shape
    total_positions = M * N
    current_knights = np.count_nonzero(board == 1)
    max_knights = total_positions // 5  # aproximación al máximo número de caballos
    remaining_knights = max_knights - current_knights

    available_positions = np.count_nonzero(board == 0)
    heuristic = min(available_positions, remaining_knights)

    return heuristic

def heuristic_3(board):
    M, N = board.shape
    available_positions = 0
    for i in range(M):
        for j in range(N):
            if board[i, j] == 0 and valid_movements(i, j, board):
                available_positions += 1
    threatened_positions = np.count_nonzero(board == -1)
    heuristic = available_positions + threatened_positions
    return heuristic

def heuristic_4(board):
    M, N = board.shape
    heuristic = 0
    if len(board)!=0:
      for i in range(M):
          for j in range(N):
              if board[i, j] == 0:
                  for mov in possible_movements:
                      x, y = i + mov[0], j + mov[1]
                      if in_board(board, x, y) and board[x, y] == 0:
                          heuristic += 1
      return heuristic
    return 0

def heuristic_5(board):
    M, N = board.shape
    heuristic = 0
    for i in range(M):
        for j in range(N):
            if board[i, j] == 0 or board[i, j] == -1:
                heuristic += 1
    return heuristic

def heuristic_6(board):
    total_knights = np.count_nonzero(board == 1)  # Contamos los caballos ya colocados
    # Asumimos que el número máximo de caballos posibles es el tamaño del tablero
    board_size = board.shape[0] * board.shape[1]
    heuristic = board_size - total_knights  # Espacio total menos caballos actuales
    return heuristic

def heuristic_7(board):
  num_knights = np.count_nonzero(board == 1)
  potential_knight_positions = 0
  for i in range(board.shape[0]):
      for j in range(board.shape[1]):
          if board[i, j] == 0 and valid_movements(i, j, board):
              potential_knight_positions += 1
  num_attacked_cells = np.count_nonzero(board == -1)
  heuristic = num_attacked_cells - potential_knight_positions - num_knights
  if is_solution(board):
    heuristic = 0
  return heuristic

def heuristic_8(board):
    M, N = board.shape
    #mat = np.fromfunction(lambda x, y: (x + y) % 2 == 0, (M, N), dtype=int)
    esquinas = np.sum(board[0, 0][board[0, 0] == 1]) + np.sum(board[0, N-1][board[0, N-1] == 1]) + np.sum(board[M-1, 0][board[M-1, 0] == 1]) + np.sum(board[M-1, N-1][board[M-1, N-1] == 1])
    diagonal = np.einsum('ii->i', board)
    suma_unos_diagonal = np.sum(diagonal[diagonal == 1])
    heuristic = -esquinas + 4 - suma_unos_diagonal + N
    return heuristic

def heuristic_8(board):
    M, N = board.shape # Obtiene las dimensiones del tablero: M es el número de filas y N el número de columnas.

    threatened_cells = np.count_nonzero(board == -1) # Cuenta la cantidad de casillas amenazadas en el tablero (representadas por -1).
    free_cells = np.count_nonzero(board == 0) # Cuenta la cantidad de casillas libres en el tablero (representadas por 0).
    knight_cells = np.count_nonzero(board == 1) # Cuenta la cantidad de casillas ocupadas por caballos en el tablero (representadas por 1).

    # Calcula el valor heurístico. Este se basa en una fórmula que pondera el total de
    # casillas amenazadas y libres frente a las ocupadas por caballos, normalizado
    # por el área total del tablero sumado al número de caballos.
    heuristic = (threatened_cells + free_cells - knight_cells) / ((M * N) + knight_cells)

    # Asegura que el valor heurístico no sea negativo, ya que esto no tendría sentido en este contexto.
    if heuristic < 0:
        heuristic = 0

    return heuristic # Retorna el valor heurístico final, que representa cuán favorable es la configuración actual del tablero.

  # Penalizo tener muchas amenazadas y libres y favorecen el número de caballos
  # Contra más caballos estas más cerca de la solución y contra más libres estas más lejos de la solución,
  # Y contra más casillas amenazadas igual que con las libres.

def prune(path_list):
    # DataFrame que almacena cada camino, el estado final (como tupla) y el coste acumulado
    paths_data = []

    for path in path_list:
        final_state = tuple(path[-1].flatten())  # Convertimos el estado final en una tupla para poder usarla en agrupación
        path_cost = cost(path)  # Calcula el coste acumulado del camino
        paths_data.append({'path': path, 'state': final_state, 'cost': path_cost})

    # Convertimos paths_data en DataFrame
    df_paths = pd.DataFrame(paths_data)

    # Ordenamos por 'cost' ascendente para tener los menores costes arriba
    df_paths.sort_values(by='cost', inplace=True)

    # Eliminamos duplicados en 'state' quedándonos solo con el de menor coste para cada estado final
    pruned_paths_df = df_paths.drop_duplicates(subset='state', keep='first')

    # Convertimos los caminos resultantes a lista de listas
    pruned_paths = pruned_paths_df['path'].tolist()

    return pruned_paths



def order_astar(old_paths, new_paths, c, h, *args, **kwargs):
    all_paths = old_paths + new_paths
    sorted_paths = sorted(all_paths, key=lambda path: c(path) + h(path[-1]))
    # Devuelve la lista de caminos ordenada y podada según A*
    return prune(sorted_paths)


def order_byb(old_paths, new_paths, c, *args, **kwargs):
    all_paths = old_paths + new_paths
    sorted_paths = sorted(all_paths, key=lambda path: c(path))
    # Ordena la lista de caminos segun el coste
    return prune(sorted_paths) # Devuelve la lista de caminos ordenada y podada segun B&B

def search(initial_board, expansion, cost, heuristic, ordering, solution):
    paths = [ [initial_board] ]
    sol = None

    while paths and sol is None:
        path = paths[0]
        board = path[-1]

        if solution(board):
          sol = board
        else:
            paths.pop(0)
            new_boards = expansion(board)
            new_paths = []
            for board in new_boards:
                new_path = path.copy()
                new_path.append(board)
                new_paths.append(new_path)
            paths = ordering(paths, new_paths, cost, heuristic)
            print(f"Number of paths remaining: {len(paths)}")
            print(board)
            print("Coste:", cost(path), "Heuristica:", {heuristic(board) if heuristic else None})

    if len(paths) > 0:
        return sol # Devuelve solo la solucion, no el camino solucion
    else:
        return None




import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("Execution time: ", end - start, " seconds")
        return res
    return wrapper

@timer
def search_horse_byb(initial_board):
    return search(initial_board, expand, cost, None, order_byb, is_solution)

@timer
def search_horse_astar(initial_board, heuristic):
    return search(initial_board, expand, cost, heuristic, order_astar, is_solution)

CONF = {'2x2': (2, 2),
        '3x3': (3, 3),
        '3x5': (3, 5),
        '5x5': (5, 5),
        '8x8': (8, 8)}

def measure_solution(board):
    res = 0
    if board is not None:  # Verificamos si el tablero no es None
      res = np.count_nonzero(board == 1)
    return res


def launch_experiment(configuration, heuristic=None):
    conf = CONF[configuration]
    print(f"Running {'A*' if heuristic else 'B&B'} with {configuration} board")
    if heuristic:
        sol = search_horse_astar(initial_state(*conf), heuristic)
    else:
        sol = search_horse_byb(initial_state(*conf))
    n_c = measure_solution(sol)
    print(f"Solution found: \n{sol}")
    print(f"Number of horses in solution: {n_c}")

    return sol, n_c

#launch_experiment('3x3')
print()

launch_experiment('2x2', heuristic=heuristic_1)
launch_experiment('3x3', heuristic=heuristic_1)
launch_experiment('3x5', heuristic=heuristic_1)
launch_experiment('5x5')
#launch_experiment('8x8', heuristic=heuristic_1)
'''

launch_experiment('2x2')
launch_experiment('3x3')
launch_experiment('3x5')
#launch_experiment('5x5')
#launch_experiment('8x8')
'''
print("Execution finished")

