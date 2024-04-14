import sys
import re
import yaml
import random
from glob import glob
from collections import defaultdict
from fractions import Fraction
import argparse


# Bresenham's line algorithm from Rosetta Code
# https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#Not_relying_on_floats
def line(xy0, xy1):
	y0, x0 = xy0
	y1, x1 = xy1
	if x0==x1 and y0==y1:
		return [] # [ (x1,y1),]
	res = []
	rev = reversed
	if abs(y1 - y0) <= abs(x1 - x0):
		x0, y0, x1, y1 = y0, x0, y1, x1
		rev = lambda x: x
	if x1 < x0:
		x0, y0, x1, y1 = x1, y1, x0, y0
	leny = abs(y1 - y0)
	return [tuple(rev((round(Fraction(i, leny) * (x1 - x0)) + x0, (1 if y1 > y0 else -1) * i + y0))) for i in range(leny + 1)]

def flood_shape(cells, start):
	seen = set()
	active = [start,]
	while active:
		cur = active.pop()
		seen.add(cur)
		for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
			new_cell = (cur[0]+dx,cur[1]+dy)
			if new_cell in cells and new_cell not in seen: 
				active.append( (cur[0]+dx,cur[1]+dy) )
	return seen

def try_shift(left, right, shift=1):
	shifted = set([(x+shift,y) for x,y in left])
	return len(shifted&right)==0

def best_shift(left, right, padding, max_shift):
	working_shifts = [0, ]
	for p in range(1,max_shift+1):
		if try_shift(left, right, shift=p):
			working_shifts.append( p )
		else: 
			break
	if len(working_shifts)>=padding:
		return working_shifts[-1-padding]
	return 0

def squeeze_space(space, padding, max_shift):
	collected_shifts = defaultdict(int)
	xses = list(sorted(set(map(lambda x:x[0], space.keys()))))
	ranges = []
	cur_range = None
	for x in xses:
		if not cur_range or x>max(cur_range)+1:
			if cur_range: ranges.append( cur_range )
			cur_range = []
		cur_range.append( x )
	if cur_range: ranges.append( cur_range )
	done = set()
	for r in ranges:
		cells_in_cur_range = set()
		for x,y in space.keys():
			if x in r:
				cells_in_cur_range.add( (x,y) )
		while cells_in_cur_range:
			start = list(sorted(cells_in_cur_range))[0]
			flooded = flood_shape(cells_in_cur_range, start)
			cells_in_cur_range -= flooded
			done |= flooded
			if cells_in_cur_range - done:
				shift = best_shift(done, cells_in_cur_range - done, padding, max_shift)
				if shift>0:
					new_space = defaultdict(str)
					for pos, mark in space.items():
						if pos in done:
							new_pos = (pos[0]+shift, pos[1])
						else:
							new_pos = pos
						new_space[new_pos] = mark
					return new_space, True
	return space, False

def draw_space(space, config=None):
	if config and config.get('fsq',-1)>=0:
		repeat = True
		while repeat:
			space, repeat = squeeze_space(space, config.get('fsq',-1), 2*config.get('w') )
	if space:
		for y in range(max(map(lambda x:x[1],space.keys()))+1,min(map(lambda x:x[1],space.keys()))-2,-1):
			row = []
			for x in range(min(map(lambda x:x[0],space.keys()))-1,max(map(lambda x:x[0],space.keys()))+2):
				row.append( space.get((x,y)," ") )
			print("".join(row))

def primitives2pixels(space, anchors2points, edges, config):
	for edge in edges:
		for dot in line(anchors2points[edge[0]],anchors2points[edge[1]]):
			for dx in range(config.get('dx',1)):
				for dy in range(config.get('dy',1)):
					space[(dot[0]+dx,dot[1]+dy)] = config.get('e','X')
	for anchor in anchors2points.values():
		for dx in range(config.get('dx',1)):
			for dy in range(config.get('dy',1)):
				space[(anchor[0]+dx,anchor[1]+dy)] = config.get('a','#')
	return space

def put_text_plain(space,text,config,geometry):
	anchors2points = dict()
	edges = []
	shift = 0	
	for i,c in enumerate(text.upper()):
		shape = geometry.get(c,geometry[' '])
		if shape['anchors']:
			for anchor, anchor_pos in shape['anchors'].items():
				x = shift+anchor_pos[0]*config['w']
				y = config[anchor_pos[1]] # config['h']-
				anchors2points["%d_%s"%(i,anchor)] = (x,y)
		if shape['edges']:
			for edge in shape['edges']:
				edges.append( ("%d_%s"%(i,edge[0]), "%d_%s"%(i,edge[1]), None, edge[2]) )
		if shape['anchors']:
			shift += max([x[0] for x in shape['anchors'].values()])*config['w']
		shift += config.get('pad',0)
	return primitives2pixels(space, anchors2points, edges, config)

def put_text_greedy(space,text, config, geometry):
	anchors2points = dict()
	edges = []
	shift = 0
	last_taken = [i for i in range(config['h']+1)]

	for i,c in enumerate(text.upper()):
		la2ga = dict()
		if c == '~':
			last_taken = [i for i in range(config['h']+1)]
			continue
		shape = geometry.get(c,geometry[' '])
		if not shape['anchors']:
			if c == ' ':
				shift += config.get('pad',0)
			continue
		left_anchors = [ (anchor[0],anchor[1][0],anchor[1][1]) for anchor in shape['anchors'].items() if anchor[1][0] == 0]
		left_anchors_pos = dict([(anchor[0],anchor[1][1]) for anchor in shape['anchors'].items() if anchor[1][0] == 0])
		left_edges   = [edge for edge in shape['edges'] if edge[0] in left_anchors_pos and edge[1] in left_anchors_pos]
		found = False
		for py in range(config['h']-config['f']):
			for my in range(config['h'],py+config['f'],-1):
				a2p = dict([(a,(0,py+(config[y]*(my-py))//config['h'])) for a,y in left_anchors_pos.items()])
				subspace = primitives2pixels(defaultdict(str), a2p, left_edges, config)
				taken = [key[1] for key in subspace.keys()]
				if not set(taken)&set(last_taken):
					found = True
					break
			if found:
				break
		if not found:
			py = 0
			my = config['h']

		right_column = max([x[0] for x in shape['anchors'].values()])
		right_anchors = set()
		for anchor, anchor_pos  in shape['anchors'].items():
			x = shift+anchor_pos[0]*config['w']
			if not found: x += config.get('pad',0)
			if not anchor_pos[0]:
				y = py+(config[anchor_pos[1]]*(my-py))//config['h']
			else:
				y = config[anchor_pos[1]]
				broken = False
				for edge in shape['edges']:
					if edge[0] == anchor and edge[1] in la2ga:
						ly = config[anchor_pos[1]]
						ry = anchors2points[la2ga[edge[1]]][1]
					elif edge[1] == anchor and edge[0] in la2ga:
						ry = config[anchor_pos[1]]
						ly = anchors2points[la2ga[edge[0]]][1]
					else:
						continue
					if edge[2] == '=' and ly != ry:
						broken = True
					elif edge[2] == '<' and ly >= ry:
						broken = True
					elif edge[2] == '<=' and ly > ry:
						broken = True
					elif edge[2] == '>' and ly <= ry:
						broken = True
					elif edge[2] == '>=' and ly < ry:
						broken = True
					if broken:
						break
				if broken: 
					y = py+(config[anchor_pos[1]]*(my-py))//config['h'] # config['h']-
			anchors2points["%d_%s"%(i,anchor)] = (x,y)
			la2ga[anchor] = "%d_%s"%(i,anchor)
			if anchor_pos[0] == right_column:
				right_anchors.add("%d_%s"%(i,anchor))
		right_edges = []
		for edge in shape['edges']:
			edges.append( ("%d_%s"%(i,edge[0]), "%d_%s"%(i,edge[1]), None, edge[2]) )
			if edges[-1][0] in right_anchors and edges[-1][1] in right_anchors:
				right_edges.append( edges[-1] )
		subspace = primitives2pixels(
										defaultdict(str), 
										dict([ item for item in anchors2points.items() if item[0] in right_anchors]), 
										right_edges, 
										config
									)
		taken = [key[1] for key in subspace.keys()]
		last_taken = taken[:]
		for i in taken:
			for j in range(-config['vc'],config['vc']+1):
				last_taken.append(i+j)
		shift += right_column*config['w']
		if not found: 
			shift += config.get('pad',0)
	return primitives2pixels(space, anchors2points, edges, config)

def pre_render_vert(anchors, edges, config, low_y, high_y):
	# print(anchors)
	anchors = dict([(a,(0,low_y+y*(high_y-low_y)//config['h'])) for a,y in anchors.items()])
	bolder_config = dict(config)
	bolder_config['dx'] += config['vc']
	bolder_config['dy'] += config['vc']
	subspace = primitives2pixels(defaultdict(str), anchors, edges, bolder_config)
	taken = list(sorted(set([key[1] for key in sorted(subspace.keys())])))
	return taken

def pre_render_field(anchors, edges, config, shift_x = 0, shift_y = 0):
	anchors = dict([(a,(pos[0]+shift_x,pos[1]+shift_y)) for a,pos in anchors.items()])
	bolder_config = dict(config)
	bolder_config['dx'] += config['vc']
	bolder_config['dy'] += config['vc']
	subspace = primitives2pixels(defaultdict(str), anchors, edges, bolder_config)
	taken = set( subspace.keys() )
	return taken

def rename_anchor(a,iteration,text,right=False):
	q = a
	if right: text = "r_"+text
	if "_" in q: q = q.split('_',1)[1]
	q = f'{iteration}_{text}_{q}'
	return q

def check_equations(matched, left_item, right_item, left_item_right_anchors, right_item_left_anchors, config):
	left_item_edge_anchors_y = {}
	left_broken = False
	if matched and matched[0] != (0, config['h']):
		# check if we can distort an edge column without resizing full left item
		low_y, high_y = matched[0]
		for a,y in left_item_right_anchors.items():
			left_item_edge_anchors_y[a] = low_y + y*(high_y - low_y)/config['h']
		broken = False
		for edge in left_item['shape']['edges']:
			if edge[1] in left_item_edge_anchors_y:
				ly = left_item['shape']['anchors'][edge[0]][1]
				ry = left_item_edge_anchors_y[edge[1]]
			elif edge[0] in left_item_edge_anchors_y:
				ry = left_item_edge_anchors_y[edge[0]]
				ly = left_item['shape']['anchors'][edge[1]][1]
			else:
				continue
			if edge[2] == '=' and ly != ry:
				broken = True
			elif edge[2] == '<' and ly >= ry:
				broken = True
			elif edge[2] == '<=' and ly > ry:
				broken = True
			elif edge[2] == '>' and ly <= ry:
				broken = True
			elif edge[2] == '>=' and ly < ry:
				broken = True
			if broken:
				break
		left_broken = broken
	right_item_edge_anchors_y = {}
	right_broken = False
	if matched and matched[1] != (0, config['h']):
		# check if we can distort an edge column without resizing full right item
		low_y, high_y = matched[1]
		for a,y in right_item_left_anchors.items():
			right_item_edge_anchors_y[a] = low_y + y*(high_y - low_y)/config['h']
		broken = False
		for edge in right_item['shape']['edges']:
			if edge[1][-2] == edge[0][-2]: continue
			if edge[1] in right_item_edge_anchors_y:
				ry = right_item['shape']['anchors'][edge[0]][1]
				ly = right_item_edge_anchors_y[edge[1]]
			elif edge[0] in right_item_edge_anchors_y:
				ly = right_item_edge_anchors_y[edge[0]]
				ry = right_item['shape']['anchors'][edge[1]][1]
			else:
				continue
			if edge[2] == '=' and ly != ry:
				broken = True
			elif edge[2] == '<' and ly >= ry:
				broken = True
			elif edge[2] == '<=' and ly > ry:
				broken = True
			elif edge[2] == '>' and ly <= ry:
				broken = True
			elif edge[2] == '>=' and ly < ry:
				broken = True
			if broken:
				break
		right_broken = broken
	return left_broken, right_broken, left_item_edge_anchors_y, right_item_edge_anchors_y

def merge_items(left_item, right_item, iteration, config):
	if left_item['text'] in " ~":
		result = right_item
		result['text'] = left_item['text']+result['text']
		return result
	if right_item['text'] in " ~":
		result = left_item
		result['text'] = result['text']+right_item['text']
		return result
	matched = False
	right_item_left_column = min([x[0] for x in right_item['shape']['anchors'].values()])
	left_item_right_column = max([x[0] for x in left_item['shape']['anchors'].values()])
	right_item_left_anchors = dict([
										(anchor[0],anchor[1][1]) 
										for anchor in right_item['shape']['anchors'].items() 
										if anchor[1][0] == right_item_left_column
									])
	right_item_left_edges   = [
								edge for edge in right_item['shape']['edges'] 
								if edge[0] in right_item_left_anchors and edge[1] in right_item_left_anchors
							]
	left_item_right_anchors = dict([
										(anchor[0],anchor[1][1]) 
										for anchor in left_item['shape']['anchors'].items() 
										if anchor[1][0] == left_item_right_column
									])
	left_item_right_edges   = [
								edge for edge in left_item['shape']['edges'] 
								if edge[0] in left_item_right_anchors and edge[1] in left_item_right_anchors
							]
	if left_item['text'][-1] not in " ~" and right_item['text'][0] not in " ~":
		left_mappers = dict()
		right_mappers = dict()
		for low_y in range(config['f']):
			for high_y in range(config['h'],config['h']-config['f'],-1):	
				_left_broken, _right_broken, _, _ = check_equations(
																		((low_y, high_y),(low_y, high_y)), 
																		left_item, right_item, 
																		left_item_right_anchors, 
																		right_item_left_anchors, 
																		config
																	)
				_left_distortion = 1.
				_resize_coef = (high_y - low_y)/config['h']
				for d in left_item['distortion_vector']:
					if _left_broken:
						_left_distortion *= d*_resize_coef
					else:
						_left_distortion *= d
				if not _left_broken:
					_left_distortion *= _resize_coef
				_right_distortion = 1.
				for d in right_item['distortion_vector']:
					if _right_broken:
						_right_distortion *= d*_resize_coef
					else:
						_right_distortion *= d
				if not _right_broken:
					_right_distortion *= _resize_coef
				left_mappers[(low_y, high_y)] = ( 
					pre_render_vert(left_item_right_anchors, left_item_right_edges, config, low_y, high_y), 
					_left_distortion, (high_y - low_y)/config['h']
				)
				right_mappers[(low_y, high_y)] = ( 
					pre_render_vert(right_item_left_anchors, right_item_left_edges, config, low_y, high_y), 
					_right_distortion, (high_y - low_y)/config['h']
				)

		matches = defaultdict(list)
		for lo, lv in left_mappers.items():
			for ro, rv in right_mappers.items():
				if not(set(lv[0])&set(rv[0])):
					matches[lv[1]*rv[1]].append( (lo, ro, lv[2], rv[2]) )

		if matches:
			best_distortion = max(matches)
			matches = matches[best_distortion]
			matched = random.choice(matches)

	right_item_shift = left_item_right_column
	if not matched:
		right_item_shift += config['pad']
		best_distortion = 1.
		matched = ((0, config['h']), (0, config['h']), 1., 1.)
		if left_item['text'][-1] == " " or right_item['text'][0] == " ":
			right_item_shift += config['w']

	left_broken, right_broken, left_item_edge_anchors_y, right_item_edge_anchors_y = check_equations(
			matched, left_item, right_item, left_item_right_anchors, right_item_left_anchors, config
		)

	matched = list(matched)
	if not left_broken: matched[2] = 1.
	left_distortion = list(map(lambda x:x*matched[2], left_item['distortion_vector']))
	if not right_broken: matched[3] = 1.
	right_distortion = list(map(lambda x:x*matched[3], right_item['distortion_vector']))

	result_anchors = dict()
	result_edges = list()
	low_y, high_y = matched[0]
	for a,v in left_item['shape']['anchors'].items():
		if not left_broken and a not in left_item_edge_anchors_y:
			result_anchors[rename_anchor(a,iteration,left_item['text'])] = v
		else:
			result_anchors[rename_anchor(a,iteration,left_item['text'])] = (
					v[0], low_y+v[1]*(high_y-low_y)//config['h']
			)
	for e in left_item['shape']['edges']:
		result_edges.append( 
			(
				rename_anchor(e[0],iteration,left_item['text']),
				rename_anchor(e[1],iteration,left_item['text']),
				e[2] 
			)
		)
	left_part = pre_render_field(result_anchors,result_edges,config)

	result_anchors_right = dict()
	result_edges_right = list()

	low_y, high_y = matched[1]
	for a,v in right_item['shape']['anchors'].items():
		if not right_broken and a not in right_item_edge_anchors_y:
			result_anchors_right[rename_anchor(a,iteration,right_item['text'],right=True)] = (right_item_shift+v[0], v[1])
		else:
			result_anchors_right[rename_anchor(a,iteration,right_item['text'],right=True)] = (
				right_item_shift+v[0], low_y+v[1]*(high_y-low_y)//config['h']
			)
	for e in right_item['shape']['edges']:
		result_edges_right.append( 
			(
					rename_anchor(e[0],iteration,right_item['text'],right=True),
					rename_anchor(e[1],iteration,right_item['text'],right=True),
					e[2] 
			)
		)
	for pad in range(-config['w']*config['sq'],1):
		if not left_part&pre_render_field(result_anchors_right,result_edges_right,config,shift_x=pad):
			# print(pad)
			break
	for a,v in result_anchors_right.items():
		result_anchors[a] = (v[0]+pad,v[1])
	result_edges.extend( result_edges_right )
	return {
				'text':left_item['text']+right_item['text'],
				'shape':{'anchors':result_anchors, 'edges':result_edges},
				'distortion_vector': left_distortion+right_distortion
			}

def unfold_shape(shape, config, prefix = ""):
	if shape['anchors']:
		shape['anchors'] = dict(
			[
				(prefix+anchor[0],(config['w']*anchor[1][0],config[anchor[1][1]])) 
				for anchor in shape['anchors'].items()
			]
		)
	if shape['edges']:
		shape['edges'] = [(prefix+e[0],prefix+e[1],e[2]) for e in shape['edges']]
	return shape

def put_text_random(space,text, config, geometry, return_raw = False):
	items = []
	new_text = ""
	for i,c in enumerate(text.upper()):
		if c in geometry:
			new_text += c
		else:
			new_text += " "
	text = re.sub(r' +', ' ', new_text.strip())
	for i,c in enumerate(text.upper()):
		items.append( {
						'text':c, 
						'shape':unfold_shape(
									dict( geometry.get(c,geometry[' ']) ),
									config, 
									prefix = f'{i}_'
								), 
						'distortion_vector':[1.,]
					} )

	iteration = 0
	while len(items)>1:
		idx = random.randint(1,len(items)-1)
		left_item  = items[idx-1]
		right_item = items[idx]
		items = items[:idx-1] + [merge_items(left_item, right_item, iteration, config),] + items[idx+1:]
		iteration += 1
	if return_raw:
		return items[0]
	return primitives2pixels(space, items[0]['shape']['anchors'], items[0]['shape']['edges'], config)

def put_text_random_best(space,text, config, geometry, attempts=None):
	best_score = None
	best_result = None
	if attempts is None:
		attempts = config.get('rtr',16)
	for _ in range(attempts):
		res = put_text_random(space,text, config, geometry, return_raw = True)
		width = max(map(lambda x:x[0],res['shape']['anchors'].values()))/len(text)
		distortion = 1.
		for d in res['distortion_vector']:
			distortion *= d
		if best_score is None or best_score<width*pow(distortion,config.get('dp',1.0)):
			best_score = width*pow(distortion,config.get('dp',1.0))
			best_result = res.copy()
	return primitives2pixels(
								defaultdict(str), 
								best_result['shape']['anchors'], 
								best_result['shape']['edges'], 
								config
							)


methods = [
	('plain', put_text_plain),
	('greedy', put_text_greedy),
	('random', put_text_random),
	('random_best', put_text_random_best)
]

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', '-c', type=str, default='cfg.yaml',
	                    help='config file')
	parser.add_argument('--geometry', '-g', type=str, default='geometry.yaml',
	                    help='font geometry file')
	parser.add_argument('--text', '-t', type=str, default='text',
	                    help='text to draw')
	parser.add_argument('--seed', '-s', type=int, default=-1,
	                    help='random seed')
	parser.add_argument('--method', '-m', type=str, default='random_best',
	                    help='method to use')
	args = parser.parse_args()

	if args.seed != -1: random.seed(args.seed)
	if args.config == 'shuf':
		args.config = random.choice( glob('cfg*.yaml') )
	if args.method == 'shuf':
		args.method = random.choice( ['greedy', 'random', 'random_best'] )
	config = yaml.load(open(args.config, encoding='utf-8').read(), Loader=yaml.FullLoader)
	geometry  = yaml.load(open(args.geometry, encoding='utf-8').read(), Loader=yaml.FullLoader)

	if args.method in dict(methods):
		space = dict(methods)[args.method](defaultdict(str), args.text, config, geometry)
		draw_space(space, config)
	else:
		print('Select one of the existing algorithms:')
		for p,fn in methods:
			print(f'Method: {p}')
			space = fn(defaultdict(str), args.text, config, geometry)
			draw_space(space, config)

if __name__ == '__main__':
    main()