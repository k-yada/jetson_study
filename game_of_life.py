#!/usr/bin/env python3

import numpy
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import curses
from curses import wrapper
import time

from pycuda.compiler import SourceModule

BLOCKSIZE = 32
GPU_NITER = 1

row2str = lambda row: ''.join("O" if c != 0 else ' ' for c in row)
cell_value = lambda world, height, width, y, x: world[y % height, x % width]

def print_world(stdscr, world, generation, elapsed):
	height, width = world.shape
	for y in range(height):
		row = world[y]
		stdscr.addstr(y, 0, row2str(row))
	stdscr.addstr(height, 0, "Generation: %06d, Elapsed: %.6f[sec]" % (generation, elapsed / generation), curses.A_REVERSE)
	stdscr.refresh()

def set_next_cell_value(world, next_world, height, width, y, x):
	current_value = cell_value(world, height, width, y, x)
	next_value = current_value
	num_live = 0
	num_live += cell_value(world, height, width, y - 1, x - 1)
	num_live += cell_value(world, height, width, y - 1, x    )
	num_live += cell_value(world, height, width, y - 1, x + 1)
	num_live += cell_value(world, height, width, y, x - 1)
	num_live += cell_value(world, height, width, y, x + 1)
	num_live += cell_value(world, height, width, y + 1, x - 1)
	num_live += cell_value(world, height, width, y + 1, x    )
	num_live += cell_value(world, height, width, y + 1, x + 1)

	if current_value == 0 and num_live == 3:
		next_value = 1
	elif current_value == 1 and num_live in (2, 3):
		next_value = 1
	else:
		next_value = 0
		
	next_world[y, x] = next_value
	#if num_live >= 4 and num_live <= 1:
	#	next_value = 0
	#else:
	#	next_value = 1

def calc_next_world_cpu(world, next_world):
	height, width = world.shape
	for y in range(height):
		for x in range(width):
			set_next_cell_value(world, next_world, height, width, y, x)

def calc_next_world_gpu(world, next_world):
	height, width = world.shape
	mod = SourceModule("""
	__global__ void calc_next_world_cpu(const int* __restrict__ world, const int* __restrict__ world, const int world_size_x, int world_size_y) {
	}
	""")
	calc_next_world_gpu = mod.get_function("calc_next_world_cpu")
	block = (BLOCKSIZE, BLOCKSIZE, 1)
	grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])
	print("Grid = ({0}, {1}), Block = ({2}, {3})".format(grid[0], grid[1], block[0], block[1]))

	start = cuda.Event()
	end = cuda.Event()
	start.record()
	for i in range(GPU_NITER):
		calc_next_world_gpu(cuda.In(world), cuda.Out(next_world), numpy.int32(width), numpy.int32(height), block = block, grid = grid)
	end.record()
	end.synchronize()

def game_of_life(stdscr, height, width):
	# 世界の初期値
	world = numpy.random.randint(2, size=(height,width), dtype=numpy.int32)

	# 次の世代の世界を保持する２事件配列
	next_world = numpy.empty((height, width), dtype=numpy.int32)

	elapsed = 0.0
	generation = 0
	while True:
		generation += 1
		print_world(stdscr, world, generation, elapsed)
		start_time = time.time()
		#calc_next_world_cpu(world, next_world)
		calc_next_world_gpu(world, next_world)
		duration = time.time() - start_time
		elapsed += duration
		world, next_world = next_world, world

def main(stdscr):
	stdscr.clear()
	stdscr.nodelay(True)
	scr_height, scr_width = stdscr.getmaxyx()
	game_of_life(stdscr, scr_height - 1, scr_width)

if __name__ == '__main__':
	curses.wrapper(main)
