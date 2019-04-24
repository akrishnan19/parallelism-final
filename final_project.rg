import "regent"

local c = regentlib.c
local m = terralib.includec("math.h")

fspace Matrix {
	elem	:	double;
}

task init_matrix(A	:	region(Matrix),
				 b	:	region(Matrix),
				 n	:	uint64)
where
	writes(A, b)
do
	
end

task toplevel()
	var n : uint64 = 1000
	var max_iter : uint64 = 5000
	var tol : double = .00001
	var A = region(ispace(ptr, n * n), Matrix)
	var b = region(ispace(ptr, n), Matrix)
	var x = region(ispace(ptr, n), Matrix)

	init_matrix(A, b, n)
end


regentlib.start(toplevel)