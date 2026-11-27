"""Генераторы рисунка стен. Они создают только входные данные симулятора."""
from collections import deque

DIRECTIONS = ((2, 0), (-2, 0), (0, 2), (0, -2))
FAMILIES = ("dfs", "prim", "braid", "rooms", "field", "blocks")
SIZES = (11, 15, 21, 31, 41)


def generate(family, side, rng, *, sparse=False, height=None):
    h, w = height or side, side
    grid = [bytearray(b"#"*w) for _ in range(h)]
    nodes = [(x, y) for y in range(1, h-1, 2) for x in range(1, w-1, 2)]

    def neighbors(p):
        x, y = p
        return [(x+dx, y+dy) for dx, dy in DIRECTIONS
                if 0 < x+dx < w-1 and 0 < y+dy < h-1]

    def join(p, q):
        x, y = q
        grid[y][x] = 46
        grid[(p[1]+y)//2][(p[0]+x)//2] = 46

    start = rng.choice(nodes)
    grid[start[1]][start[0]] = 46
    visited = {start}
    if family == "wilson":
        order = nodes.copy()
        rng.shuffle(order)
        for p in order:
            if p in visited:
                continue
            links = {}
            while p not in visited:
                q = rng.choice(neighbors(p))
                links[p] = q
                p = q
            p = next(iter(links))
            while p not in visited:
                visited.add(p)
                q = links[p]
                join(p, q)
                grid[p[1]][p[0]] = 46
                p = q
    elif family == "division":
        for y in range(1,h-1):
            grid[y][1:w-1] = b"."*(w-2)
        rooms = [(0,0,w-1,h-1)]
        while rooms:
            x0,y0,x1,y1 = rooms.pop()
            if min(x1-x0, y1-y0) < 4:
                continue
            vertical = (x1-x0 > y1-y0 or (x1-x0 == y1-y0 and rng.randrange(2)))
            if vertical:
                wall, door = rng.randrange(x0+2,x1,2), rng.randrange(y0+1,y1,2)
                for y in range(y0+1,y1):
                    grid[y][wall] = 46 if y == door else 35
                rooms.extend(((x0,y0,wall,y1),(wall,y0,x1,y1)))
            else:
                wall, door = rng.randrange(y0+2,y1,2), rng.randrange(x0+1,x1,2)
                for x in range(x0+1,x1):
                    grid[wall][x] = 46 if x == door else 35
                rooms.extend(((x0,y0,x1,wall),(x0,wall,x1,y1)))
    elif family == "prim":
        edges = [(start,q) for q in neighbors(start)]
        while edges:
            i = rng.randrange(len(edges))
            edges[i],edges[-1] = edges[-1],edges[i]
            p,q = edges.pop()
            if q not in visited:
                visited.add(q)
                join(p,q)
                edges.extend((q,r) for r in neighbors(q) if r not in visited)
    elif family == "field":
        density = rng.uniform(.17,.39)
        for y in range(1,h-1):
            for x in range(1,w-1):
                if rng.random() > density:
                    grid[y][x] = 46
        # Оставляем одну четырёхсвязную область.
        queue = deque([start])
        while queue:
            x,y = queue.popleft()
            for q in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if grid[q[1]][q[0]] == 46 and q not in visited:
                    visited.add(q)
                    queue.append(q)
        for y in range(1,h-1):
            for x in range(1,w-1):
                if (x,y) not in visited:
                    grid[y][x] = 35
    else:
        # Для блоков алгоритм сперва связывает клетки внутри одного района.
        # К соседнему району переходит через редкие ворота.
        stack = [start]
        block = rng.choice((4,6,8))
        while stack:
            p = stack[-1]
            choices = [q for q in neighbors(p) if q not in visited]
            if not choices:
                stack.pop()
                continue
            if family == "blocks":
                within = [q for q in choices if (q[0]//block,q[1]//block) == (p[0]//block,p[1]//block)]
                choices = within or choices
            q = rng.choice(choices)
            join(p,q)
            visited.add(q)
            stack.append(q)
        if family == "braid":
            for x,y in nodes:
                choices = [q for q in neighbors((x,y)) if grid[(y+q[1])//2][(x+q[0])//2] == 35]
                if choices and rng.random() < (.10 if sparse else .38):
                    join((x,y),rng.choice(choices))
        if family == "rooms":
            for _ in range(max(2,w*h//95)):
                x,y = rng.choice(nodes)
                a,b = rng.randint(2,6),rng.randint(2,6)
                for j in range(y,min(y+b,h-1)):
                    grid[j][x:min(x+a,w-1)] = b"."*(min(x+a,w-1)-x)
    return tuple(row.decode("ascii") for row in grid)
