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
--  __demand(__vectorize)
  for i in is do
    x[i] = 0.0
    b[i] = 1.0
    r[i] = 1.0
    r_prev[i] = 1.0
    p[i] = 0.0
    s[i] = 0.0
  end
end

task inner_product1(v1     : region(ispace(int1d), double))
where 
  reads (v1)
do
  var sum = 0.0
  for i in v1.ispace do
    sum += v1[i] * v1[i]
  end

  return sum
end

task inner_product2(v1     : region(ispace(int1d), double),
                    v2     : region(ispace(int1d), double))
where 
  reads (v1, v2)
do
  var sum = 0.0
  for i in v1.ispace do
    sum += v1[i] * v2[i]
  end

  return sum
end

task CG_dir(r        : region(ispace(int1d),double),
            r_prev   : region(ispace(int1d),double),
            p        : region(ispace(int1d),double),
            beta     : double)
where 
  reads(r), reads writes(r_prev, p)
do
--  __demand(__vectorize)
  for i in r.ispace do
    p[i]  = r[i] + beta*p[i]
    r_prev[i] = r[i] 
  end
end

task CG_mtv(is       : ispace(int1d),
            is_nnz   : ispace(int1d),
            A        : region(Entry),
            p        : region(is,double),
            s        : region(is,double))
where 
  reads(A.{a,ci,ri}, p), reads writes(s)
do
  for i in is_nnz do
    s[A[i].ri] += A[i].a * p[A[i].ci]
--    c.printf("for entry %i, (%.2f, %i, %i)\n", i, A[i].a, A[i].ri, A[i].ci)
  end
end

task CG_iter(x        : region(ispace(int1d),double),
             r        : region(ispace(int1d),double),
             r_prev   : region(ispace(int1d),double),
             p        : region(ispace(int1d),double),
             s        : region(ispace(int1d),double),
             alpha    : double)
where 
  reads(p, r_prev), reads writes(x, r, s)
do
--  __demand(__vectorize)
  for i in x.ispace do
    x[i] += alpha*p[i]
  end

--  __demand(__vectorize)
  for i in r.ispace do
    r[i] = r_prev[i] - alpha*s[i]
  end

--  __demand(__vectorize)
  for i in s.ispace do
    s[i] = 0.0
  end
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
  c.printf("* Parallelism : %11u *\n",   config.parallelism)
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
  var temp1 = 0.0
  var temp2 = 0.0

  -- Creating equal vector partitions
  var c0 = ispace(int1d, config.parallelism)

  var px = partition(equal, x, c0)
  var pb = partition(equal, b, c0)
  var pr = partition(equal, r, c0)
  var pr_prev = partition(equal, r_prev, c0)
  var pp = partition(equal, p, c0)
  var ps = partition(equal, s, c0)

  -- Creating inner product temp vars
--  var temp1 = region(c0, double)
--  var temp2 = region(c0, double)

  -- Initialize the page graph from a file
  initialize_matrix(A, is_int, config.matrix_order, config.nnz)
  initialize_vectors(is, x, b, r, r_prev, p, s, config.matrix_order)

  var num_iterations = 0
  var converged = false
  var ts_start = c.legion_get_current_time_in_micros()
  while not converged do
    num_iterations += 1

    temp1 = 0.0
    temp2 = 0.0
    __demand(__parallel)
    for c in c0 do
      temp1 += inner_product1(pr[c])
    end

    __demand(__parallel)
    for c in c0 do
      temp2 += inner_product1(pr_prev[c])
    end
    beta = temp1 / temp2

    __demand(__parallel)
    for c in c0 do
      CG_dir(pr[c], pr_prev[c], pp[c], beta)
    end

    CG_mtv(is, is_nnz, A, p, s)

    temp1 = 0.0
    temp2 = 0.0
    __demand(__parallel)
    for c in c0 do
      temp1 += inner_product1(pr_prev[c])

    end

    __demand(__parallel)
    for c in c0 do
      temp2 += inner_product2(pp[c], ps[c])
    end
    alpha = temp1 / temp2

    __demand(__parallel)
    for c in c0 do
      CG_iter(px[c], pr[c], pr_prev[c], pp[c], ps[c], alpha)
    end  

    err = 0.0
    for c in c0 do
      err += inner_product1(pr[c])
    end

    c.printf("----- ERROR IS: %11.4f ------\n", err)
    if (err < config.error_bound) then
      converged = true
    end
  end
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Conjugate gradient converged after %d iterations in %.4f sec\n",
    num_iterations, (ts_stop - ts_start) * 1e-6)
--  for i in is do
--    c.printf("x[%i] is: %.2f\n", i, x[i])
--  end
end

regentlib.start(toplevel)
