import fastfilters as ff
import numpy as np
import sys

a = np.random.randn(1000000).reshape(1000,1000)
k = ff.FIRKernel(0, 1.0)

full = ff.convolve_fir(a, [k,k])

tests = [
	((200,200), (400,400)),
	((200,300), (400,400)),
	((0,10), (400,400)),
	((900,900), (990,990))
]


for roi in tests:
	start,stop = roi
	x0,y0 = start
	x1,y1 = stop


	part = ff.convolve_fir(a, [k,k], roi=roi)
	part_gt = full[y0:y1,x0:x1]

	print(roi, " --> ", np.max(part - part_gt))
	if not np.allclose(part, part_gt):
		raise Exception("ROI result != full result")


with open(sys.argv[1], 'w') as f:
	f.write('')
