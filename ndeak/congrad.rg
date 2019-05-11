import "regent"

-- Helper module to handle command line arguments
local PageRankConfig = require("congrad_config")

local c = regentlib.c

fspace Entry {
  a         : double;
  ci        : uint64;
  ri        : uint64;
}


task initialize_matrix(A        : region(Entry),
                       is       : ispace(int1d),
                       ord      : uint64,
                       nnz      : uint64)
where
  reads writes(A.a, A.ci, A.ri)
do
  -- Filling in interior row entries
  var A_idx = 3
  var idx = 1
  for i in is do
    A[A_idx - 1].a = 1.0
    A[A_idx].a     =-2.0
    A[A_idx + 1].a = 1.0

    A[A_idx - 1].ci = idx-1
    A[A_idx].ci     = idx
    A[A_idx + 1].ci = idx+1

    A[A_idx - 1].ri = idx
    A[A_idx].ri     = idx
    A[A_idx + 1].ri = idx

    A_idx += 3
    idx += 1
  end

  -- Filling boundary conditions
  A[0].a = -2.0
  A[1].a = 1.0
  
  A[0].ci = 0
  A[1].ci = 1

  A[0].ri = 0
  A[1].ri = 0
  
  A[nnz-1].a = -2.0
  A[nnz-2].a = 1.0

  A[nnz-1].ci = ord-1
  A[nnz-2].ci = ord-2

  A[nnz-1].ri = ord-1
  A[nnz-2].ri = ord-1
end

task initialize_vectors(is       : ispace(int1d),
                        x        : region(is,double),
                        b        : region(is,double),
                        r        : region(is,double),
                        r_prev   : region(is,double),
                        p        : region(is,double),
                        s        : region(is,double),
                        ord      : uint64)
where 
  reads writes (x, b, r, r_prev, p, s)
do
  for i in is do
    x[i] = 0.0
    b[i] = 1.0
    r[i] = 1.0
    r_prev[i] = 1.0
    p[i] = 0.0
    s[i] = 0.0
  end
end

task inner_product(is     : ispace(int1d),
                   v1     : region(is, double),
                   v2     : region(is, double))
where 
  reads(v1, v2)
do
  var val : double = 0.0
  for i in is do
    val += v1[i] * v2[i]
  end
  return val
end

task vector_copy()

end

task CG_iter(is       : ispace(int1d),
             is_nnz   : ispace(int1d),
             A        : region(Entry),
             x        : region(is,double),
             b        : region(is,double),
             r        : region(is,double),
             r_prev   : region(is,double),
             p        : region(is,double),
             s        : region(is,double),
             alpha    : double,
             beta     : double,
             ord      : uint64,
             nnz      : uint64)
where 
  reads(A.{a,ci,ri}, b), reads writes(x, b, r, r_prev, p, s)
do
  beta = inner_product(is, r, r) / inner_product(is, r_prev, r_prev)
--  c.printf("value of beta is: %.2f, should be 1.0\n", beta)
  for i in is do
    p[i]  =r[i] + beta*p[i]
    r_prev[i] = r[i] 
--    c.printf("p[%i] is %.2f\n", i, p[i])
  end

  for i in is_nnz do
    s[A[i].ri] += A[i].a * p[A[i].ci]
--    c.printf("for entry %i, (%.2f, %i, %i)\n", i, A[i].a, A[i].ri, A[i].ci)
  end

  alpha = inner_product(is, r_prev, r_prev) / inner_product(is, p, s)
--  c.printf("alpha is: %.2f\n", alpha)
  for i in is do
--    c.printf("s[%i] is %.2f\n", i, s[i])
    x[i] += alpha*p[i]
    r[i] = r_prev[i] - alpha*s[i]
    s[i] = 0.0
  end

  var err = inner_product(is, r, r)

  return err
end

task toplevel()
  var config : ConGradConfig
  config:initialize_from_command()
  c.printf("**********************************\n")
  c.printf("* PageRank                       *\n")
  c.printf("*                                *\n")
  c.printf("* Error Bound      : %11g *\n",   config.error_bound)
  c.printf("* Max # Iterations : %11u *\n",   config.max_iterations)
  c.printf("* Matrix Order : %11u *\n",   config.matrix_order)
  c.printf("**********************************\n")

  -- Creating index spaces for problem iterations
  var is = ispace(int1d, config.matrix_order)
  var is_int = ispace(int1d, config.matrix_order - 2)
  var is_nnz = ispace(int1d, config.nnz)

  -- Create a region for our matrix A
  var A = region(ispace(ptr, config.nnz, 0), Entry)

  -- Creating regions for solution and RHS vectors
  var x = region(is, double)
  var b = region(is, double)
  
  -- Creating residual vectors
  var r = region(is, double)
  var r_prev = region(is, double)
  
  -- Creating temporary vectors for CG iterations
  var p = region(is, double)
  var s = region(is, double)

  -- Creating other temporary variables used in CG iterations
  var alpha = 0.0
  var beta = 0.0
  var err = 1000.0

  -- Initialize the page graph from a file
  initialize_matrix(A, is_int, config.matrix_order, config.nnz)
  initialize_vectors(is, x, b, r, r_prev, p, s, config.matrix_order)

  var num_iterations = 0
  var converged = false
  var ts_start = c.legion_get_current_time_in_micros()
  while not converged do
    num_iterations += 1
    err = CG_iter(is, is_nnz, A, x, b, r, r_prev, p, s, alpha, beta, config.matrix_order, config.nnz)
    -- c.printf("----- ERROR IS: %11.4f ------\n", err)
    if (err < config.error_bound) then
      converged = true
    end
  end
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Conjugate gradient converged after %d iterations in %.4f sec\n",
    num_iterations, (ts_stop - ts_start) * 1e-6)
  for i in is do
    c.printf("x[%i] is: %.2f\n", i, x[i])
  end
end

regentlib.start(toplevel)
