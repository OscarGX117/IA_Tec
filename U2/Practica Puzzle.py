from queue import PriorityQueue

# Definir el estado inicial y el estado objetivo del rompecabezas
initial_state = [
    [7, 2, 4],
    [5, 0, 6],
    [8, 3, 1]
]

goal_state = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
]

# Función para encontrar la posición de un número en el estado actual
def find_number(state, number):
    for i in range(3):
        for j in range(3):
            if state[i][j] == number:
                return j, i  # Intercambiamos las coordenadas

# Función para encontrar la posición del espacio vacío en el estado actual
def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return j, i  # Intercambiamos las coordenadas

# Función para generar los movimientos posibles a partir del estado actual
def generate_moves(state):
    moves = []
    x, y = find_blank(state)
    
    # Movimiento hacia arriba
    if y > 0:
        new_state = [row[:] for row in state]
        new_state[y][x], new_state[y - 1][x] = new_state[y - 1][x], new_state[y][x]
        moves.append(new_state)
    
    # Movimiento hacia abajo
    if y < 2:
        new_state = [row[:] for row in state]
        new_state[y][x], new_state[y + 1][x] = new_state[y + 1][x], new_state[y][x]
        moves.append(new_state)
    
    # Movimiento hacia la izquierda
    if x > 0:
        new_state = [row[:] for row in state]
        new_state[y][x], new_state[y][x - 1] = new_state[y][x - 1], new_state[y][x]
        moves.append(new_state)
    
    # Movimiento hacia la derecha
    if x < 2:
        new_state = [row[:] for row in state]
        new_state[y][x], new_state[y][x + 1] = new_state[y][x + 1], new_state[y][x]
        moves.append(new_state)
    
    return moves

# Función para calcular la heurística (número de casillas fuera de lugar)
def heuristic(state):
    count = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != goal_state[i][j]:
                count += 1
    return count

# Función para resolver el rompecabezas usando el algoritmo A*
def solve_puzzle(initial_state):
    open_list = PriorityQueue()
    open_list.put((heuristic(initial_state), initial_state))
    came_from = {}
    g_score = {tuple(map(tuple, initial_state)): 0}
    
    while not open_list.empty():
        _, current_state = open_list.get()
        
        if current_state == goal_state:
            # Reconstruir el camino desde el estado objetivo hasta el estado inicial
            path = []
            while current_state != initial_state:
                path.append(current_state)
                current_state = came_from[tuple(map(tuple, current_state))]
            path.append(initial_state)
            return path[::-1]
        
        for next_state in generate_moves(current_state):
            tentative_g_score = g_score[tuple(map(tuple, current_state))] + 1
            if tuple(map(tuple, next_state)) not in g_score or tentative_g_score < g_score[tuple(map(tuple, next_state))]:
                g_score[tuple(map(tuple, next_state))] = tentative_g_score
                f_score = tentative_g_score + heuristic(next_state)
                open_list.put((f_score, next_state))
                came_from[tuple(map(tuple, next_state))] = current_state
    
    return None

# Resolver el rompecabezas
solution = solve_puzzle(initial_state)

if solution:
    for i, state in enumerate(solution):
        print(f"Paso {i + 1}:")
        for row in state:
            print(row)
        number_7_x, number_7_y = find_number(state, 7)  # Intercambiamos las coordenadas
        print(f"Posición del número 7: X={number_7_x}, Y={number_7_y}")
        print()
    print(f"El costo fue de: {len(solution) - 1} movimientos.")
else:
    print("No se encontró solución.")

